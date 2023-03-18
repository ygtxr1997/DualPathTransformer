import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.utils_load_from_cfg import instantiate_from_config


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
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()
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
        x = self.layer4(x)

        return x


class TokenPooling(nn.Module):
    def __init__(self, num_tokens: int, dim: int,
                 pool_type: str = 'conv',
                 pool_scale: int = 2,):
        super(TokenPooling, self).__init__()
        height = int(np.sqrt(num_tokens))
        assert height * height == num_tokens

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.token_to_d4 = Rearrange('b (h w) c -> b c h w', h=height)
        self.pool = nn.Sequential(
                conv1x1(dim, dim * pool_scale, pool_scale),
                nn.BatchNorm2d(dim * pool_scale, eps=1e-05, )
            )
        self.d4_to_token = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        x = self.norm(x)
        x = self.token_to_d4(x).contiguous()
        x = self.pool(x)
        x = self.d4_to_token(x).contiguous()
        return x


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


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class StyleVectorizer(nn.Module):
    def __init__(self, dim_in, dim_emb, depth, heads: int, lr_mul=1.0):
        super().__init__()

        layers = []
        for i in range(depth):
            if i == 0:
                layers.extend([EqualLinear(dim_in, dim_emb, lr_mul), leaky_relu()])
            else:
                layers.extend([EqualLinear(dim_emb, dim_emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)
        self.out_heads = heads

    def forward(self, x):
        x = rearrange(x, 'b h n d -> b n (h d)').contiguous()
        x = F.normalize(x, dim=1)
        x = self.net(x)
        x = rearrange(x, 'b n (h d) -> b h n d', h=self.out_heads).contiguous()
        return x


class MlpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads: int):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )
        self.out_heads = heads

    def forward(self, x):
        """
        :param x: (b,h,n1,d)
        :return: (b,h,n1,d)
        """
        x = rearrange(x, 'b h n d -> b n (h d)').contiguous()
        x = F.normalize(x, dim=1)
        x = self.mlp(x)
        x = rearrange(x, 'b n (h d) -> b h n d', h=self.out_heads).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 opn_inner_dim=6*64):
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

        ''' dpt_sub '''
        # self.mlp_ck = MlpBlock(opn_inner_dim, inner_dim, heads)
        # self.mlp_cv = MlpBlock(opn_inner_dim, inner_dim, heads)
        self.mlp_ck = StyleVectorizer(opn_inner_dim, inner_dim, 3, heads)  # v6
        self.mlp_cv = StyleVectorizer(opn_inner_dim, inner_dim, 3, heads)  # v6

        # self.sub_ratio = nn.Parameter(torch.randn(size=(1, heads, 1, 1)) + 1)
        # self.sub_ratio = nn.Parameter(torch.randn(size=(1, heads, 1, dim_head)) + 1)  # v4
        # self.sub_ratio = 1.  # v5
        self.sub_ratio = nn.Parameter(torch.randn(size=(1, heads, 1, dim_head)) + 1)  # v6

        # self.nonlinear_c = nn.GELU()  # v4
        # self.nonlinear_c = MlpBlock(inner_dim, inner_dim, heads)  # v5
        self.nonlinear_c = MlpBlock(inner_dim, inner_dim, heads)  # v6

    def forward(self, x, ck=None, cv=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # (b,h,n1,d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b,h,n1,n1)

        attn = self.attend(dots.contiguous())
        out = torch.matmul(attn, v)  # (b,h,n1,d)

        if ck is not None and cv is not None:
            ck = self.mlp_ck(ck)
            cv = self.mlp_cv(cv)
            dots_c = torch.matmul(q, ck.transpose(-1, -2)) * self.scale
            attn_c = self.attend(dots_c.contiguous())
            out_c = torch.matmul(attn_c, cv)  # (b,h,n1,d)
            out_c = self.nonlinear_c(out_c)
            out = out - out_c * self.sub_ratio
            # if np.random.randint(1000) < 2 and x.device == torch.device(0):
            #     print('ratio', self.sub_ratio.mean(), self.sub_ratio.min(), self.sub_ratio.max())

        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
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
        self.depth = depth
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x, ck=None, cv=None):
        for idx, (sa1, ff1) in enumerate(self.layers):
            if idx < -1:
                x = sa1(x) + x
            else:
                x = sa1(x, ck=ck, cv=cv) + x
            x = ff1(x) + x
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FaceTransformerBackbone(nn.Module):
    def __init__(self,
                 cnn_layers,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 pattern_dim=256,
                 feature_dim=512,
                 use_cls_token=True,
                 fp16=False):
        super(FaceTransformerBackbone, self).__init__()
        self.fp16 = fp16

        self.extractor_id = IResBackbone(IBasicBlock, cnn_layers)

        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.token_id = nn.Parameter(torch.randn(1, 1, dim))

        # self.pe_id = nn.Parameter(torch.randn(1, 14*14+1, dim))
        height = 14 // (pattern_dim // 256)
        pos_emb = get_2d_sincos_pos_embed(pattern_dim, height).reshape((height, height, pattern_dim))
        pos_emb = torch.FloatTensor(pos_emb).unsqueeze(0)
        pos_emb = rearrange(pos_emb, 'b h w c -> b c h w').contiguous()
        self.register_buffer('pe_id', pos_emb.contiguous())

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        num_tokens = height * height
        if self.use_cls_token:
            num_tokens += 1
        self.id_to_out = nn.Sequential(
            nn.LayerNorm(num_tokens * dim, eps=1e-05),
        )

        self.fc = nn.Linear(num_tokens * dim, feature_dim)  # TODO: fix feature size to 512
        self.features = nn.BatchNorm1d(feature_dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

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

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)

            # position embedding
            pat_id += self.pe_id

            emb_id = self.to_patch_embedding_id(pat_id)
            b, n, _ = emb_id.shape

            if self.use_cls_token:
                # Embedding[:, 0, :] insert token
                tokens_id = repeat(self.token_id, '() n d -> b n d', b=b)
                emb_id = torch.cat((tokens_id, emb_id), dim=1)  # (b, n+1, d)
            emb_id = self.dropout(emb_id)

            # Transformer
            emb_id = self.transformer(emb_id)
            emb_id = torch.flatten(emb_id, 1)  # (B,37632)

            # emb_id = emb_id[:, 0]
            emb_id = self.id_to_out(emb_id)  # [-4, 4], norm=22

        """ op1. vit """
        # emb_id = emb_id.float() if self.fp16 else emb_id
        # emb_id = self.id_to_out(emb_id)
        """ op2. arcface """
        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)
        return emb_id


class FaceVitBackbone(nn.Module):
    def __init__(self,
                 start_channel: int,
                 early_depths: int,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 feature_dim=512,
                 use_cls_token=True,
                 cls_token_nums: int = 1,
                 fp16=False):
        super(FaceVitBackbone, self).__init__()
        self.fp16 = fp16

        self.up_sample = 1
        self.extractor_id = EarlyConv(start_channel=start_channel, depths=early_depths, up_sample=1)
        double_channel_times = min(early_depths, 4)
        pattern_dim = start_channel * (2 ** (double_channel_times - 1))

        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        self.use_cls_token = use_cls_token
        self.cls_token_nums = cls_token_nums
        if self.use_cls_token:
            self.token_id = nn.Parameter(torch.randn(1, cls_token_nums, dim))

        # self.pe_id = nn.Parameter(torch.randn(1, 14*14+1, dim))
        height = 112 // (2 ** 3)  # down_sample - up_sample == 3
        pos_emb = get_2d_sincos_pos_embed(pattern_dim, height).reshape((height, height, pattern_dim))
        pos_emb = torch.FloatTensor(pos_emb).unsqueeze(0)
        pos_emb = rearrange(pos_emb, 'b h w c -> b c h w').contiguous()
        self.register_buffer('pe_id', pos_emb.contiguous())

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        num_tokens = height * height
        if self.use_cls_token:
            num_tokens += cls_token_nums
        out_tokens = cls_token_nums if self.use_cls_token else num_tokens
        self.id_to_out = nn.Sequential(
            nn.LayerNorm(out_tokens * dim, eps=1e-05),
        )

        self.fc = nn.Linear(out_tokens * dim, feature_dim)  # TODO: fix feature size to 512
        self.features = nn.BatchNorm1d(feature_dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

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

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)

            # position embedding
            pat_id += self.pe_id

            emb_id = self.to_patch_embedding_id(pat_id)
            b, n, _ = emb_id.shape

            if self.use_cls_token:
                # Embedding[:, 0, :] insert token
                tokens_id = repeat(self.token_id, '() n d -> b n d', b=b)
                emb_id = torch.cat((tokens_id, emb_id), dim=1)  # (b, t+n, d)
            emb_id = self.dropout(emb_id)

            # Transformer
            emb_id = self.transformer(emb_id)
            if self.use_cls_token:
                emb_id = emb_id[:, :self.cls_token_nums]  # (B,t,192)
                emb_id = torch.flatten(emb_id, 1)  # (B,t,192)->(B,t*192)
            else:
                emb_id = torch.flatten(emb_id, 1)  # (B,14*14,192)->(B,37632)
            emb_id = self.id_to_out(emb_id)  # [-4, 4], norm=22

        """ op1. vit """
        # eid.float(mb_id = emb_) if self.fp16 else emb_id
        # emb_id = self.id_to_out(emb_id)
        """ op2. arcface """
        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)
        return emb_id


class FacePoolTransformerBackbone(nn.Module):
    def __init__(self,
                 cnn_layers,
                 dim,
                 depths,
                 heads,
                 mlp_dim,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 pattern_dim=256,
                 feature_dim=512,
                 fp16=False):
        super(FacePoolTransformerBackbone, self).__init__()
        self.fp16 = fp16

        self.extractor_id = IResBackbone(IBasicBlock, cnn_layers)

        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        height = 14 // (pattern_dim // 256)

        self.stages = nn.ModuleList()
        for idx, depth in enumerate(depths):

            num_tokens = height * height

            pos_emb = get_2d_sincos_pos_embed(dim, height).reshape((height, height, dim))
            pos_emb = torch.FloatTensor(pos_emb).unsqueeze(0)
            pos_emb = rearrange(pos_emb, 'b h w c -> b (h w) c', h=height).contiguous()
            self.register_buffer('pe_id_%d' % idx, pos_emb)

            emb_drop = nn.Dropout(emb_dropout)
            transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            pool = nn.Identity()
            if idx < len(depths) - 1:
                pool = TokenPooling(num_tokens, dim, pool_type='conv', pool_scale=2)
                height //= 2
                dim *= 2

            self.stages.append(torch.nn.Sequential(
                emb_drop, transformer, pool
            ))

        num_tokens = height * height
        self.id_to_out = nn.Sequential(
            nn.LayerNorm(num_tokens * dim, eps=1e-05),
            nn.BatchNorm1d(num_tokens * dim, eps=1e-05)
        )

        self.fc = nn.Linear(num_tokens * dim, feature_dim)
        self.features = nn.BatchNorm1d(feature_dim, eps=1e-05)  # layernorm is bad
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)

            emb_id = self.to_patch_embedding_id(pat_id)
            b, n, _ = emb_id.shape

            for idx, stage in enumerate(self.stages):
                # position embedding
                pe_id = getattr(self, 'pe_id_%d' % idx)
                emb_id += pe_id
                emb_id = stage(emb_id)

            emb_id = torch.flatten(emb_id, 1)  # (B,H*H*C)

            emb_id = self.id_to_out(emb_id)  # [-4, 4], norm=22

        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)
        return emb_id


class EarlyConv(nn.Module):
    def __init__(self,
                 start_channel: int = 24,
                 depths: int = 4,
                 up_sample: int = 2,
                 ):
        super(EarlyConv, self).__init__()
        self.start_channel = start_channel
        in_channels, out_channels = [], []
        strides = []
        if depths <= 4:
            in_channels = [3, start_channel, start_channel * 2, start_channel * 4]
            out_channels = [start_channel, start_channel * 2, start_channel * 4, start_channel * 8]
            if up_sample == 2:
                strides = [2, 2, 2, 2]
            elif up_sample == 1:
                strides = [1, 2, 2, 2]
        elif depths == 6:
            out_channels = [start_channel, start_channel,
                            start_channel, start_channel * 2,
                            start_channel * 4, start_channel * 8]
            in_channels = [3] + out_channels[:-1]
            if up_sample == 2:
                strides = [1, 2, 1, 2, 2, 2]
            elif up_sample == 1:
                strides = [1, 1, 1, 2, 2, 2]
        else:
            raise ValueError('EarlyConv does not support depths=%d' % depths)
        self.up_sample = up_sample

        self.layers = nn.ModuleList()
        for idx in range(depths):
            layer = nn.Sequential(
                nn.Conv2d(in_channels[idx], out_channels[idx],
                          kernel_size=3, stride=strides[idx], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[idx], eps=1e-05),
                nn.PReLU(out_channels[idx])
            )
            self.layers.append(layer)
        self.layers.append(nn.Conv2d(out_channels[depths - 1], out_channels[depths - 1],
                                     kernel_size=1, stride=1, padding=0, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.up_sample > 1:
            x = F.interpolate(x, scale_factor=self.up_sample, mode='bicubic', align_corners=True)
        for layer in self.layers:
            x = layer(x)
        return x


class FaceEarlyTransformerBackbone(nn.Module):
    def __init__(self,
                 start_channel: int,
                 early_depths: int,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 emb_dropout=0.,
                 dim_head=64,
                 dropout=0.,
                 feature_dim=512,
                 use_cls_token=False,
                 fp16=False):
        super(FaceEarlyTransformerBackbone, self).__init__()
        self.fp16 = fp16

        self.up_sample = 1
        self.extractor_id = EarlyConv(start_channel=start_channel, depths=early_depths, up_sample=1)
        double_channel_times = min(early_depths, 4)
        pattern_dim = start_channel * (2 ** (double_channel_times - 1))

        self.to_patch_embedding_id = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(pattern_dim, dim)
        )

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.token_id = nn.Parameter(torch.randn(1, 1, dim))

        # self.pe_id = nn.Parameter(torch.randn(1, 14*14+1, dim))
        height = 112 // (2 ** 3)  # down_sample - up_sample == 3
        pos_emb = get_2d_sincos_pos_embed(pattern_dim, height).reshape((height, height, pattern_dim))
        pos_emb = torch.FloatTensor(pos_emb).unsqueeze(0)
        pos_emb = rearrange(pos_emb, 'b h w c -> b c h w').contiguous()
        self.register_buffer('pe_id', pos_emb.contiguous())

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        num_tokens = height * height
        if self.use_cls_token:
            num_tokens += 1
        self.id_to_out = nn.Sequential(
            nn.LayerNorm(num_tokens * dim, eps=1e-05),
        )

        self.fc = nn.Linear(num_tokens * dim, feature_dim)  # TODO: fix feature size to 512
        self.features = nn.BatchNorm1d(feature_dim, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

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

    def forward(self, x, ck=None, cv=None):
        with torch.cuda.amp.autocast(self.fp16):
            # CNN
            pat_id = self.extractor_id(x)

            # position embedding
            pat_id += self.pe_id

            emb_id = self.to_patch_embedding_id(pat_id)
            b, n, _ = emb_id.shape

            if self.use_cls_token:
                # Embedding[:, 0, :] insert token
                tokens_id = repeat(self.token_id, '() n d -> b n d', b=b)
                emb_id = torch.cat((tokens_id, emb_id), dim=1)  # (b, n+1, d)
            emb_id = self.dropout(emb_id)

            # Transformer
            emb_id = self.transformer(emb_id, ck, cv)
            emb_id = torch.flatten(emb_id, 1)  # (B,37632)

            # emb_id = emb_id[:, 0]
            emb_id = self.id_to_out(emb_id)  # [-4, 4], norm=22

        """ op1. vit """
        # emb_id = emb_id.float() if self.fp16 else emb_id
        # emb_id = self.id_to_out(emb_id)
        """ op2. arcface """
        emb_id = emb_id.float() if self.fp16 else emb_id
        emb_id = self.fc(emb_id)
        emb_id = self.features(emb_id)
        return emb_id


class FaceTransformerHeader(nn.Module):
    def __init__(self,
                 header_type: str,
                 header_num_classes: int,
                 header_params_m: float,
                 header_params_s: float = 64.0,
                 header_params_a: float = 0.,
                 header_params_k: float = 1.0,
                 ):
        super(FaceTransformerHeader, self).__init__()
        feature_dim = 512
        from tricks.margin_losses import AMCosFace, Softmax, AMArcFace
        header_type = header_type.lower()
        if 'cosface' in header_type:
            self.loss = AMCosFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'arcface' in header_type:
            self.loss = AMArcFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'softmax' in header_type:
            self.loss = Softmax(in_features=feature_dim,
                                out_features=header_num_classes,
                                device_id=None, )
        else:
            raise ValueError('Header type not supported.')

    def forward(self, v, label=None):
        if self.training:
            final = self.loss(v, label)
            return final  # id:(b, dim)
        else:
            return v


class FaceTransformerWithHeader(nn.Module):
    def __init__(self,
                 header_type: str,
                 header_num_classes: int,
                 header_params_m: float,
                 header_params_s: float = 64.0,
                 header_params_a: float = 0.,
                 header_params_k: float = 1.0,
                 backbone_config: dict = None):
        super(FaceTransformerWithHeader, self).__init__()
        self.backbone = instantiate_from_config(backbone_config)

        feature_dim = 512
        from tricks.margin_losses import AMCosFace, Softmax, AMArcFace
        header_type = header_type.lower()
        if 'cosface' in header_type:
            self.loss = AMCosFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'arcface' in header_type:
            self.loss = AMArcFace(in_features=feature_dim,
                                  out_features=header_num_classes,
                                  device_id=None,
                                  m=header_params_m, s=header_params_s,
                                  a=header_params_a, k=header_params_k)
        elif 'softmax' in header_type:
            self.loss = Softmax(in_features=feature_dim,
                                out_features=header_num_classes,
                                device_id=None,)
        else:
            raise ValueError('Header type not supported.')

    def forward(self, x, label=None):
        emb_id = self.backbone(x)
        if self.training:
            final = self.loss(emb_id, label)
            return final  # id:(b, dim)
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
