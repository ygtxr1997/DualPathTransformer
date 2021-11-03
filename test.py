import torch

import config as cfg
from thop import profile

img = torch.randn(1, 3, 112, 112)

""" 1. Dual-Path Transformer """
from backbone.dual_path_transformer import DualPathTransformer
dpt = DualPathTransformer(cnn_layers=[2, 2, 2],
                          dim=cfg.dp_set.dim,
                          depth=cfg.dp_set.depth,
                          heads_id=cfg.dp_set.heads_id,
                          heads_oc=cfg.dp_set.heads_oc,
                          mlp_dim_id=cfg.dp_set.mlp_dim_id,
                          mlp_dim_oc=cfg.dp_set.mlp_dim_oc,
                          num_classes=93431,
                          dim_head_id=cfg.dp_set.dim_head_id,
                          dim_head_oc=cfg.dp_set.dim_head_oc,
                          dropout_oc=0.2,
                          fp16=False).cuda()
# feature_id, feature_oc = dpt(img.cuda())
# print(feature_id.shape, feature_oc.shape)

""" 2. Face Transformer """
from backbone.face_transformer import FaceTransformer
ft = FaceTransformer(cnn_layers=[2, 2, 2],
                     num_classes=93431,
                     dim=cfg.ft_set.dim,
                     depth=cfg.ft_set.depth,
                     heads=cfg.ft_set.heads,
                     mlp_dim=cfg.ft_set.mlp_dim,
                     emb_dropout=0.,
                     dim_head=cfg.ft_set.dim_head,
                     dropout=0.,
                     fp16=False).cuda()
# feature_id = ft(img.cuda())
# print(feature_id.shape)

""" 3. IResnet + MLP """
from backbone.iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from torch import nn
class iresnet_mlp(nn.Module):
    def __init__(self):
        super(iresnet_mlp, self).__init__()
        self.iresnet = iresnet34(False)
        self.mlp = nn.Linear(512, 93431)
    def forward(self, x):
        x = self.iresnet(x)
        x = self.mlp(x)
        return x

irnet = iresnet_mlp().cuda()

""" 4. Segment Transformer """
from backbone.seg_transformer import SegTransformer
st = SegTransformer(cnn_layers=[2, 2, 2],
                     num_classes=93431,
                     dim=512,
                     depth=1,
                     heads=4,
                     mlp_dim=256,
                     emb_dropout=0.,
                     dim_head=64,
                     dropout=0.,
                     fp16=False).cuda()

""" =================== flops&params ====================== """
import backbone

model = ft
macs, params = profile(model,
                       inputs=(img.cuda(), ),
                       custom_ops={
                           backbone.face_transformer.Attention: backbone.face_transformer.Attention.cnt_flops,
                           backbone.face_transformer.FeedForward: backbone.face_transformer.FeedForward.cnt_flops,
                           backbone.seg_transformer.Attention: backbone.seg_transformer.Attention.cnt_flops,
                           backbone.seg_transformer.FeedForward: backbone.seg_transformer.FeedForward.cnt_flops,
                           backbone.dual_path_transformer.Attention: backbone.dual_path_transformer.Attention.cnt_flops,
                           backbone.dual_path_transformer.FeedForward: backbone.dual_path_transformer.FeedForward.cnt_flops,
                           backbone.dual_path_transformer.CrossAttention: backbone.dual_path_transformer.CrossAttention.cnt_flops,

                       },)
from thop import clever_format
macs, params = clever_format([macs, params], "%.2f")
print('MACs:', macs, 'wrong:', params)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('%.2fM' % (params / 1e6))

import time
time.sleep(5)