import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from backbone.face_transformer import get_2d_sincos_pos_embed

__all__ = ['OccPerceptualNetwork']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResBackboneSeg(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 conv1_stride=1):
        super(IResBackboneSeg, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=conv1_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        if len(layers) >= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        if len(layers) >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=False)
        if len(layers) >= 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x_feats.append(x)  # x0: (64, 56, 56)

        x = self.layer1(x)
        x_feats.append(x)  # x1: (64, 28, 28)
        x = self.layer2(x)
        x_feats.append(x)  # x2: (128, 14, 14)
        x = self.layer3(x)
        x_feats.append(x)  # x3: (256, 7, 7)

        return x_feats


class PreNorm(nn.Module):
    def __init__(self,
                 dim,
                 fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def cnt_flops(m, x, y):
        x = x[0]  # (b, hw+1, dim)
        hw = x.shape[-2]
        flops = hw * 1. * m.dim * m.hidden_dim
        flops += hw * 1. * m.hidden_dim * m.dim
        # m.total_ops += flops  # no need to re-calculate


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.dim = dim
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # (b,h,n2,d)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b,h,n2,n2)

        attn = self.attend(dots.contiguous())

        out = torch.matmul(attn, v)  # (b,h,n2,d)
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        return self.to_out(out), k, v


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def _vanilla_forward(self, x):
        for idx, (sa1, ff1) in enumerate(self.layers):
            x = sa1(x) + x
            x = ff1(x) + x
        return x, None, None

    def _kv_forward(self, x):
        k, v = None, None
        for idx, (sa1, ff1) in enumerate(self.layers):
            out, k, v = sa1(x)
            x = out + x
            x = ff1(x) + x
        return x, k, v

    def forward(self, x, ret_kv: bool = False):
        return self._kv_forward(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.conv2 = Conv2dReLU(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=use_batchnorm,
        # )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        # x = self.conv2(x)
        return x


class SegmentHead(nn.Module):
    def __init__(self,
                 dim: int,
                 num_classes=2,
                 ):
        super(SegmentHead, self).__init__()

        self.dec1 = DecoderBlock(dim, 128, skip_channels=128)
        self.dec2 = DecoderBlock(128, 64, skip_channels=64)
        self.dec3 = DecoderBlock(64, 32, skip_channels=64)

        self.out_conv = conv3x3(32, num_classes)
        self.out_up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, x_feats):  # x:(dim, 7, 7)
        """
        x0: 128, 14, 14
        x1: 64, 28, 28
        x2: 64, 56, 56
        """
        x0, x1, x2 = x_feats[0], x_feats[1], x_feats[2]

        x = self.dec1(x, x0)
        x = self.dec2(x, x1)
        x = self.dec3(x, x2)

        x = self.out_conv(x)
        x = self.out_up(x)
        return x  # (2, 112, 112)


class OccPerceptualNetwork(nn.Module):
    def __init__(self,
                 cnn_layers: list,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 num_classes=2,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 fp16=False):
        super().__init__()
        self.fp16 = fp16

        conv1_stride = 2
        self.extractor_oc = IResBackboneSeg(IBasicBlock, cnn_layers, conv1_stride=conv1_stride)

        pattern_dim = 64 * 8 // conv1_stride  # default:256
        down_sample_times = len(cnn_layers)  # default:4
        self.to_patch_embedding_oc = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        # self.pe_oc = nn.Parameter(torch.randn(1, 7*7, dim))
        height = 112 // (2 ** down_sample_times)
        pos_emb = get_2d_sincos_pos_embed(pattern_dim, height).reshape((height, height, pattern_dim))
        pos_emb = torch.FloatTensor(pos_emb).unsqueeze(0)
        pos_emb = rearrange(pos_emb, 'b h w c -> b c h w').contiguous()
        self.register_buffer('pe_oc', pos_emb.contiguous())

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.oc_to_4d = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-05),
            Rearrange('b (h w) c -> b c h w', c=dim, h=height),
        )
        self.oc_head = SegmentHead(dim=dim, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, label=None, ret_kv: bool = False):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            x_feats = self.extractor_oc(x)
            pat_oc = x_feats[-1]

            # Position Embedding
            pat_oc += self.pe_oc

            emb_oc = self.to_patch_embedding_oc(pat_oc)  # projection
            b, n, _ = emb_oc.shape

            emb_oc = self.dropout(emb_oc)

            # Transformer
            emb_oc, ock, ocv = self.transformer(emb_oc, ret_kv=ret_kv)

            out_oc = self.oc_to_4d(emb_oc)

        out_oc = out_oc.float() if self.fp16 else out_oc
        skip_feats = x_feats[:3][::-1]
        out_oc = self.oc_head(out_oc, skip_feats)

        if not ret_kv:
            return out_oc
        else:
            return out_oc, ock.detach(), ocv.detach()


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
    img = torch.randn(1, 3, 112, 112)
    net = OccPerceptualNetwork(cnn_layers=[2, 2, 2, 2],
                               dim=24 * 8,
                               depth=3,
                               heads=6,
                               dim_head=64,
                               mlp_dim=256,
                               num_classes=2)
    feats = net(img)
    print(feats[0].shape)

    flops = FlopCountAnalysis(net, img)
    print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
    print("[fvcore] #Params: %.2fM" % (parameter_count(net)[''] / 1e6))
