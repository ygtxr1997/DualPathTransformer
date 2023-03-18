import torch
import torch.nn as nn

from einops import rearrange, repeat

from utils.utils_load_from_cfg import instantiate_from_config
from backbone.face_transformer import FaceEarlyTransformerBackbone
from backbone.face_transformer import FaceTransformerHeader
from backbone.occ_net import OccPerceptualNetwork


class DPTSubtract(nn.Module):
    def __init__(self,
                 ft_config,
                 ft_head_config,
                 opn_config,
                 ):
        super(DPTSubtract, self).__init__()
        self.ft_net: FaceEarlyTransformerBackbone = instantiate_from_config(ft_config)
        self.ft_header: FaceTransformerHeader = instantiate_from_config(ft_head_config)
        self.opn_net: OccPerceptualNetwork = instantiate_from_config(opn_config)
        opn_weight = torch.load(opn_config.resume, map_location="cpu")
        self.opn_net.load_state_dict(opn_weight)
        self.opn_net.requires_grad_(False)
        self.opn_net.eval()

    def forward(self, x, label=None):
        occ_k, occ_v = None, None
        with torch.no_grad():
            occ_out, occ_k, occ_v = self.opn_net(x, ret_kv=True)
        x = self.ft_net(x, occ_k, occ_v)
        x = self.ft_header(x, label)
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load('configs/dpt_sub.yaml')
    print('config is:', config)
    model: torch.nn.Module = instantiate_from_config(config.model)
    model = model.cuda()
    model.train()

    bs = 5
    img = torch.randn(5, 3, 112, 112).cuda()
    lab = torch.zeros((bs), dtype=torch.long).cuda()
    out = model(img, label=lab)
    print('train out:', out.shape)
    out.mean().backward()
    print('Backward success.')

    with torch.no_grad():
        model.eval()
        out = model(img)
        print('test out:', out.shape)
