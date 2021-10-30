import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['ft_r18']

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

class IResBackbone(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,):
        super(IResBackbone, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


from backbone.segmodel.large_kernel import _GlobalConvModule
class SegmentHead(nn.Module):
    def __init__(self,
                 num_classes=2,
                 kernel_size=7,
                 dap_k=3,):
        super(SegmentHead, self).__init__()

        self.gcm1 = _GlobalConvModule(32, num_classes * 4, (kernel_size, kernel_size))

        self.deconv1 = nn.ConvTranspose2d(num_classes * 4, num_classes * dap_k ** 2, kernel_size=3,
                                          stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)

        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k, dap_k))
        )

    def forward(self, x):  # (32, 4, 4)
        x = self.gcm1(x)  # (8, 4, 4)
        x = self.deconv1(x)  # (2*9, 7, 7)
        x = self.deconv2(x)  # (2*9, 14, 14)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.DAP(x)
        return x  # (2, 112, 112)


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
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for sa1, ff1, sa2, ff2 in self.layers:
            x = sa1(x) + x
            x = ff1(x) + x
            x = sa2(x) + x
            x = ff2(x) + x
        return x


class FaceTransformer(nn.Module):
    def __init__(self,
                 cnn_layers,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 num_classes,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 fp16=False):
        super().__init__()
        self.fp16 = fp16

        self.extractor_id = IResBackbone(IBasicBlock, cnn_layers)

        pattern_dim = 256
        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        self.token_id = nn.Parameter(torch.randn(1, 1, dim))

        self.pe_id = nn.Parameter(torch.randn(1, 14*14+1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.id_to_out = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-05),
            nn.BatchNorm1d(dim, eps=1e-05),
            # nn.Linear(dim, dim)
        )
        self.fc = nn.Linear(dim, dim)
        self.features = nn.BatchNorm1d(dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False

        # self.id_head = nn.Linear(dim, num_classes)

        from tricks.margin_losses import CosFace, Softmax, ArcFace
        self.loss = CosFace(in_features=dim, out_features=num_classes, device_id=None,
                            m=0.35, s=64.0)
        # self.loss = Softmax(in_features=dim, out_features=num_classes, device_id=None,)

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

    def forward(self, x, label=None):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)
            emb_id = self.to_patch_embedding_id(pat_id)
            b, n, _ = emb_id.shape

            # Embedding[:, 0, :] insert token
            tokens_id = repeat(self.token_id, '() n d -> b n d', b=b)
            emb_id = torch.cat((tokens_id, emb_id), dim=1)  # (b, n+1, d)

            # PE
            emb_id += self.pe_id[:, :(n + 1)]
            emb_id = self.dropout(emb_id)

            # Transformer
            emb_id = self.transformer(emb_id)

            emb_id = emb_id[:, 0]
            emb_id = self.id_to_out(emb_id)  # [-4, 4], norm=22

        """ op1. vit """
        # emb_id = emb_id.float() if self.fp16 else emb_id
        # emb_id = self.id_to_out(emb_id)
        """ op2. arcface """
        # x_copy = emb_id.detach()
        # print('emb_min:', x_copy.min(),'emb_max', x_copy.max(),'emb_mean', x_copy.mean())
        # copy_norm = x_copy * x_copy
        # norm_norm = torch.nn.functional.normalize(x_copy)
        # norm_norm = norm_norm * norm_norm
        # print('norm:', torch.sqrt(copy_norm[0].sum()),
        #       'after normal norm:', torch.sqrt(norm_norm[0].sum()),
        #       'shape', x_copy.shape)
        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)

        if self.training:
            final = self.loss(emb_id, label)
            return final  # id:(b, dim), oc:(b, 2, 112, 112)
        else:
            return emb_id


""" Conditional Positional Encodings for Vision Transformers (arXiv 2021) """
class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        # Only for demo use, more complicated functions are effective too.
    def forward(self, x, H, W):
        B, N, C = x.shape
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


def _dpt(arch, layers, **kwargs):
    model = FaceTransformer(layers, **kwargs)
    return model


def ft_r18(pretrained=False, **kwargs):
    return _dpt('dpt-tiny', [2, 2, 2], **kwargs)