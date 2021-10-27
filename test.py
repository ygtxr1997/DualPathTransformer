import torch

from backbone.dual_path_transformer import DualPathTransformer

img = torch.randn(64, 3, 112, 112)

dpt = DualPathTransformer(cnn_layers=[2, 2, 2],
                          dim=512,
                          depth=5,
                          heads=8,
                          mlp_dim=512,
                          num_classes=100000,
                          fp16=True).cuda()
feature_id, feature_oc = dpt(img.cuda())
print(feature_id.shape, feature_oc.shape)
import time

time.sleep(5)