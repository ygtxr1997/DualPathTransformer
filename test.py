import torch

import config as cfg
from thop import profile

img = torch.randn(1, 3, 112, 112)

""" 1. Dual-Path Transformer """
from backbone.dual_path_transformer import DualPathTransformer
dpt = DualPathTransformer(cnn_layers=[2, 2, 2],
                          dim=512,
                          depth=2,
                          heads=8,
                          mlp_dim=512,
                          num_classes=93431,
                          dim_head=64,
                          fp16=False).cuda()
# feature_id, feature_oc = dpt(img.cuda())
# print(feature_id.shape, feature_oc.shape)

""" 2. Face Transformer """
from backbone.face_transformer import FaceTransformer
ft = FaceTransformer(cnn_layers=[2, 2, 2],
                     num_classes=93431,
                     dim=512,
                     depth=1,
                     heads=8,
                     mlp_dim=512,
                     emb_dropout=0.,
                     dim_head=64,
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
                       },)
from thop import clever_format
macs, params = clever_format([macs, params], "%.2f")
print(macs, params)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('%.2fM' % (params / 1e6))

import time
time.sleep(5)