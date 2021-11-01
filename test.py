import torch

from backbone.dual_path_transformer import DualPathTransformer
import config as cfg

from thop import profile

img = torch.randn(1, 3, 112, 112)

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

from backbone.face_transformer import FaceTransformer

ft = FaceTransformer(cnn_layers=[2, 2, 2],
                     num_classes=93431,
                     dim=512,
                     depth=2,
                     heads=8,
                     mlp_dim=512,
                     emb_dropout=0.,
                     dim_head=64,
                     dropout=0.,
                     fp16=False).cuda()
# feature_id = ft(img.cuda())
# print(feature_id.shape)


macs, params = profile(dpt, inputs=(img.cuda(), ))
from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print(params, macs)

import time
time.sleep(5)