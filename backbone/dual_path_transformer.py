import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['dpt_r18s3_ca1', 'dpt_r34s3_ca1', 'dpt_r50s3_ca1']

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


class IResBackboneSeg(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,):
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
        x_feats = []
        x = self.conv1(x)  # get x0: (64, 112, 112)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x_feats.append(x)  # x1: (64, 56, 56)
        x = self.layer2(x)
        x_feats.append(x)  # x2: (128, 28, 28)
        x = self.layer3(x)
        x_feats.append(x)  # x3: (256, 14, 14)

        return x_feats


from backbone.segmodel.large_kernel import _GlobalConvModule
class SegmentHead(nn.Module):
    def __init__(self,
                 num_classes=2,
                 kernel_size=7,
                 dap_k=3,):
        super(SegmentHead, self).__init__()

        self.gcm1 = _GlobalConvModule(32, num_classes * 4, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(256, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(256, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(128, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm5 = _GlobalConvModule(64, num_classes * dap_k ** 2, (kernel_size, kernel_size))

        self.deconv1 = nn.ConvTranspose2d(num_classes * 4, num_classes * dap_k ** 2, kernel_size=3,
                                          stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4,
                                          stride=2, padding=1, bias=False)

        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k, dap_k))
        )

    def forward(self, x, x_feats):  # (32, 4, 4)
        assert len(x_feats) == 3
        """
        x1: 64, 56, 56
        x2: 128, 28, 28
        x3: 256, 14, 14
        """
        x1, x2, x3 = x_feats[0], x_feats[1], x_feats[2]

        x = self.gcm1(x)  # (8, 4, 4)
        x = self.deconv1(x)  # (2*9, 7, 7)

        # x3 = self.gcm2(x3)
        # x = self.deconv2(torch.cat((x, x3), 1))  # (2*9, 14, 14)
        x = self.deconv2(x)

        x3 = self.gcm3(x3)
        x = self.deconv3(torch.cat((x, x3), 1))

        x2 = self.gcm4(x2)
        x = self.deconv4(torch.cat((x, x2), 1))

        x1 = self.gcm5(x1)
        x = self.deconv5(torch.cat((x, x1), 1))

        x = self.DAP(x)
        return x  # (2, 112, 112)


class PreNorm(nn.Module):
    def __init__(self,
                 dim,
                 fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormDual(nn.Module):
    def __init__(self,
                 dim,
                 fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class PreNormCat(nn.Module):
    def __init__(self,
                 dim,
                 fn,):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        n_id = x1.shape[1]
        n_oc = x2.shape[1]
        xc = torch.cat((x1, x2), dim=1)
        xc = self.norm(xc)
        assert n_id + n_oc == xc.shape[1]
        x1, x2 = xc[:, :n_id], xc[:, n_id:]
        return self.fn(x1, x2, **kwargs)


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
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    def cnt_flops(m, x, y):
        x = x[0]  # (b, hw+1, dim)
        hw = x.shape[-2]
        flops = hw * 2. * m.dim * m.hidden_dim
        flops += hw * 2. * m.hidden_dim * m.dim
        m.total_ops += flops


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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def cnt_flops(m, x, y):
        x = x[0]  # (b, n+1, dim)
        flops = 3. * 2. * m.inner_dim * m.dim * x.shape[-2]
        flops += 2. * m.heads * (m.dim_head ** 2) * x.shape[1]
        flops += 2. * m.heads * m.dim_head * (x.shape[1] ** 2)
        flops += 2. * m.inner_dim * m.dim * y.shape[-2]
        m.total_ops += flops


class CrossAttention(nn.Module):
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
        self.to_q_id = nn.Linear(dim, inner_dim, bias=False)
        self.to_q_oc = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv_id = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_kv_oc = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out_1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.to_out_2 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        """ Gating """
        self.gate_1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Tanh(),
        )
        self.gate_2 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Tanh(),
        )

    def forward(self, x1, x2):
        x_id, x_oc = x1, x2

        q1 = self.to_q_id(x_id)
        q2 = self.to_q_oc(x_oc)

        d1 = q1.detach()
        d2 = q2.detach()
        q1 = q1 + self.gate_1(d2) * q1
        q2 = q2 + self.gate_2(d1) * q2

        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)
        kv_id = self.to_kv_id(x_id).chunk(2, dim=-1)

        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)
        kv_oc = self.to_kv_oc(x_oc).chunk(2, dim=-1)

        k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv_id)
        k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv_oc)

        """ 1. cat """
        # k_all = torch.cat((k1, k2), dim=-2)
        # v_all = torch.cat((v1, v2), dim=-2)
        #
        # dots_1 = torch.matmul(q1, k_all.transpose(-1, -2)) * self.scale
        # attn_1 = self.attend(dots_1)
        # out_1 = torch.matmul(attn_1, v_all)
        # out_1 = rearrange(out_1, 'b h n d -> b n (h d)')
        #
        # dots_2 = torch.matmul(q2, k_all.transpose(-1, -2)) * self.scale
        # attn_2 = self.attend(dots_2)
        # out_2 = torch.matmul(attn_2, v_all)
        # out_2 = rearrange(out_2, 'b h n d -> b n (h d)')

        """ 2. separate """
        dots_1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn_1 = self.attend(dots_1)
        out_1 = torch.matmul(attn_1, v1)
        out_1 = rearrange(out_1, 'b h n d -> b n (h d)')

        dots_2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn_2 = self.attend(dots_2)
        out_2 = torch.matmul(attn_2, v2)
        out_2 = rearrange(out_2, 'b h n d -> b n (h d)')

        out_1 = self.to_out_1(out_1)
        out_2 = self.to_out_2(out_2)

        return out_1, out_2

    def cnt_flops(m, x, y):
        x = x[0]  # (b, n+1, dim)
        y = y[0]
        flops = 3. * 2. * m.inner_dim * m.dim * x.shape[-2]
        flops += 2. * m.heads * (m.dim_head ** 2) * x.shape[1]
        flops += 2. * m.heads * m.dim_head * (x.shape[1] ** 2)
        flops += 2. * m.inner_dim * m.dim * y.shape[-2]
        m.total_ops += 2. * flops


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads_id,
                 heads_oc,
                 dim_head_id,
                 dim_head_oc,
                 mlp_dim_id,
                 mlp_dim_oc,
                 dropout_id=0.,
                 dropout_oc=0.,):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.ModuleList([

                # PreNorm(dim, Attention(dim, heads=heads_id, dim_head=dim_head_id, dropout=dropout_id)),
                # PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_id, dropout=dropout_id)),
                # PreNorm(dim, Attention(dim, heads=heads_oc, dim_head=dim_head_oc, dropout=dropout_oc)),
                # PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_oc, dropout=dropout_oc)),

                # PreNormCat(dim, CrossAttention(dim, heads=heads_id, dim_head=dim_head_id, dropout=dropout_id)),
                PreNormDual(dim, CrossAttention(dim, heads=heads_id, dim_head=dim_head_id, dropout=dropout_id)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_id, dropout=dropout_id)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_id, dropout=dropout_oc)),  # TODO: mlp_dim_oc

                PreNorm(dim, Attention(dim, heads=heads_id, dim_head=dim_head_id, dropout=dropout_id)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_id, dropout=dropout_id)),
                PreNorm(dim, Attention(dim, heads=heads_oc, dim_head=dim_head_oc, dropout=dropout_oc)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim_oc, dropout=dropout_oc)),
            ]))

    def forward(self, x1, x2):
        # for sa1, ff1, sa2, ff2, ca, ff3, ff4 in self.layers:
        #     x1 = sa1(x1) + x1
        #     x1 = ff1(x1) + x1
        #     x2 = sa2(x2) + x2
        #     x2 = ff2(x2) + x2
        #
        #     r1, r2 = x1, x2
        #     x1, x2 = ca(x1, x2)
        #     x1 = x1 + r1
        #     x2 = x2 + r2
        #
        #     x1 = ff3(x1) + x1
        #     x2 = ff4(x2) + x2

        for ca, ff3, ff4, sa1, ff1, sa2, ff2 in self.layers:

            r1, r2 = x1, x2
            x1, x2 = ca(x1, x2)
            x1 = x1 + r1
            x2 = x2 + r2

            x1 = ff3(x1) + x1
            x2 = ff4(x2) + x2

            x1 = sa1(x1) + x1
            x1 = ff1(x1) + x1
            x2 = sa2(x2) + x2
            x2 = ff2(x2) + x2

        # for ca, ff3, ff4 in self.layers:
        #
        #     r1, r2 = x1, x2
        #     x1, x2 = ca(x1, x2)
        #     x1 = x1 + r1
        #     x2 = x2 + r2
        #
        #     x1 = ff3(x1) + x1
        #     x2 = ff4(x2) + x2

        return x1, x2


class DualPathTransformer(nn.Module):
    def __init__(self,
                 cnn_layers,
                 dim,
                 depth,
                 heads_id,
                 heads_oc,
                 mlp_dim_id,
                 mlp_dim_oc,
                 num_classes,
                 emb_dropout=0.,
                 dim_head_id=64,
                 dim_head_oc=64,
                 dropout_id=0.,
                 dropout_oc=0.,
                 fp16=False):
        super().__init__()
        self.fp16 = fp16

        self.extractor_id = IResBackbone(IBasicBlock, cnn_layers)
        self.extractor_oc = IResBackboneSeg(IBasicBlock, [2, 2, 2])

        pattern_dim = 256
        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )
        self.to_patch_embedding_oc = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        self.token_id = nn.Parameter(torch.randn(1, 1, dim))
        self.token_oc = nn.Parameter(torch.randn(1, 1, dim))

        n_id = 14 * 14
        n_oc = 14 * 14
        self.pe_id = nn.Parameter(torch.randn(1, n_id+1, dim))
        self.pe_oc = nn.Parameter(torch.randn(1, n_oc+1, dim))

        self.type_id = nn.Parameter(torch.randn(1, 1, dim))
        self.type_oc = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth,
                                       heads_id, heads_oc,
                                       dim_head_id, dim_head_oc,
                                       mlp_dim_id, mlp_dim_oc,
                                       dropout_id, dropout_oc,)

        self.id_to_out = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-05),
            nn.BatchNorm1d(dim, eps=1e-05),
        )
        feature_dim = 512
        self.fc = nn.Linear(dim, feature_dim)
        self.features = nn.BatchNorm1d(feature_dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        from tricks.margin_losses import CosFace, Softmax, ArcFace
        self.loss = CosFace(in_features=feature_dim, out_features=num_classes, device_id=None,
                            m=0.35, s=64.0)

        self.oc_to_4d = nn.Sequential(
            nn.Linear(dim, feature_dim),
            nn.LayerNorm(feature_dim),
            Rearrange('b (c h w) -> b c h w', c=32, h=4),
        )
        self.oc_head = SegmentHead(num_classes=2, kernel_size=7, dap_k=3)
        self.id_head = nn.Linear(dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, label=None):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)
            x_feats = self.extractor_oc(x)
            pat_oc = x_feats[-1]

            emb_id = self.to_patch_embedding_id(pat_id)
            emb_oc = self.to_patch_embedding_oc(pat_oc)

            b, n, _ = emb_id.shape

            # Embedding[:, 0, :] insert token
            tokens_id = repeat(self.token_id, '() n d -> b n d', b=b)
            tokens_oc = repeat(self.token_oc, '() n d -> b n d', b=b)

            emb_id = torch.cat((tokens_id, emb_id), dim=1)  # (b, n+1, d)
            emb_oc = torch.cat((tokens_oc, emb_oc), dim=1)

            # PE
            emb_id += self.pe_id[:, :(n + 1)]
            emb_oc += self.pe_oc[:, :(n + 1)]

            # Type Embedding
            emb_id += self.type_id
            emb_oc += self.type_oc

            emb_id = self.dropout(emb_id)
            emb_oc = self.dropout(emb_oc)

            # Transformer
            emb_id, emb_oc = self.transformer(emb_id, emb_oc)

            emb_id = emb_id[:, 0]
            emb_oc = emb_oc[:, 0]

            emb_id = self.id_to_out(emb_id)
            out_oc = self.oc_to_4d(emb_oc)

        # Occ-Head & Id-Head
        out_oc = out_oc.float() if self.fp16 else out_oc
        x_float_feats = []
        for x_feat in x_feats:
            x_float_feats.append(x_feat.float() if self.fp16 else x_feat)
        out_oc = self.oc_head(out_oc, x_float_feats)

        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)

        if self.training:
            final = self.loss(emb_id, label)
            return final, out_oc  # id:(b, dim), oc:(b, 2, 112, 112)
        else:
            return emb_id


class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        """ Conditional Positional Encodings for Vision Transformers (arXiv 2021) """
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
    model = DualPathTransformer(layers, **kwargs)
    return model


def dpt_r18s3_ca1(pretrained=False, **kwargs):
    return _dpt('dpt-tiny', [2, 2, 2], **kwargs)


def dpt_r34s3_ca1(pretrained=False, **kwargs):
    return _dpt('dpt-tiny', [3, 4, 3], **kwargs)


def dpt_r50s3_ca1(pretrained=False, **kwargs):
    return _dpt('dpt-tiny', [3, 4, 3], **kwargs)


class DPTInfer(nn.Module):
    def __init__(self):
        super(DPTInfer, self).__init__()
        from easydict import EasyDict as edict

        cfg = edict()
        cfg.dataset = "ms1m-retinaface-t2"
        cfg.embedding_size = 512
        cfg.sample_rate = 1
        cfg.fp16 = True
        cfg.momentum = 0.9
        cfg.weight_decay = 5e-4
        cfg.batch_size = 128  # 128
        cfg.lr = 3e-4  # 0.1 for batch size is 512

        cfg.num_classes = 93431  # 91180

        """ Setting for Model DualPathTransformer"""
        dp_set = edict()
        dp_set.dim = 384
        dp_set.depth = 2
        dp_set.heads_id = 6
        dp_set.heads_oc = 6
        dp_set.dim_head_id = 64
        dp_set.dim_head_oc = 64
        dp_set.mlp_dim_id = 192
        dp_set.mlp_dim_oc = 192
        dp_set.emb_dropout = 0.1
        dp_set.dropout_id = 0.1
        dp_set.dropout_oc = 0.1
        cfg.model_set = dp_set

        net = dpt_r18s3_ca1(False,
                            fp16=cfg.fp16,
                            num_classes=cfg.num_classes,
                            dim=cfg.model_set.dim,
                            depth=cfg.model_set.depth,
                            heads_id=cfg.model_set.heads_id,
                            heads_oc=cfg.model_set.heads_oc,
                            mlp_dim_id=cfg.model_set.mlp_dim_id,
                            mlp_dim_oc=cfg.model_set.mlp_dim_oc,
                            emb_dropout=cfg.model_set.emb_dropout,
                            dim_head_id=cfg.model_set.dim_head_id,
                            dim_head_oc=cfg.model_set.dim_head_oc,
                            dropout_id=cfg.model_set.dropout_id,
                            dropout_oc=cfg.model_set.dropout_oc)
        net.eval()

        weight = torch.load('/gavin/code/DualPathTransformer/out/tmp_0/backbone.pth')
        net.load_state_dict(weight)
        self.net = net

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    img = torch.randn(1, 3, 112, 112)
    net = DPTInfer()
    feat = net(img)
    print('feat:', feat.shape)  # (B,256,14,14)

    import thop
    import backbone
    import thop.vision.basic_hooks
    flops, params = thop.profile(net, inputs=(img,),
                                custom_ops={
                                   backbone.face_transformer.Attention: backbone.face_transformer.Attention.cnt_flops,
                                   backbone.face_transformer.FeedForward: backbone.face_transformer.FeedForward.cnt_flops,
                                   backbone.seg_transformer.Attention: backbone.seg_transformer.Attention.cnt_flops,
                                   backbone.seg_transformer.FeedForward: backbone.seg_transformer.FeedForward.cnt_flops,
                                   backbone.dual_path_transformer.Attention: backbone.dual_path_transformer.Attention.cnt_flops,
                                   backbone.dual_path_transformer.FeedForward: backbone.dual_path_transformer.FeedForward.cnt_flops,
                                   backbone.dual_path_transformer.CrossAttention: backbone.dual_path_transformer.CrossAttention.cnt_flops,
                                   backbone.dual_path_transformer.PreNormDual: backbone.dual_path_transformer.PreNormDual.cnt_flops,
                                   backbone.dual_path_transformer_only_sa.Attention: \
                                       backbone.dual_path_transformer_only_sa.Attention.cnt_flops,
                                   backbone.dual_path_transformer_only_sa.FeedForward: \
                                       backbone.dual_path_transformer_only_sa.FeedForward.cnt_flops,
                                }, )
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
