import torch
import mxnet as mx
from mxnet import ndarray as nd
import pickle
import os
from typing import List

from eval import verification

class ReadMXNet(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        # self.highest_acc: float = 0.0
        # self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.rec_prefix = rec_prefix
        self.val_targets = val_targets
        # if self.rank is 0:
        #     self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    # def ver_test(self, backbone: torch.nn.Module, global_step: int):
    #     results = []
    #     for i in range(len(self.ver_list)):
    #         acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
    #             self.ver_list[i], backbone, 10, 10)
    #         logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
    #         logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
    #         if acc2 > self.highest_acc_list[i]:
    #             self.highest_acc_list[i] = acc2
    #         logging.info(
    #             '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
    #         results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = self.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def load_bin(self, path, image_size):
        try:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f)  # py2
        except UnicodeDecodeError as e:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        data_list = []
        # for flip in [0, 1]:
        #     data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        #     data_list.append(data)
        for idx in range(len(issame_list) * 2):
            _bin = bins[idx]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))  # (C, H, W)

            img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            import PIL.Image as Image
            fig = Image.fromarray(img.asnumpy(), mode='RGB')
            data_list.append(fig)
            # data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 1000 == 0:
                print('loading bin', idx)

            # # save img to '/home/yuange/dataset/LFW/rgb-arcface'
            # img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            # # save_name = 'ind_' + str(idx) + '.bmp'
            # # import os
            # # save_name = os.path.join('/home/yuange/dataset/LFW/rgb-arcface', save_name)
            # import PIL.Image as Image
            # fig = Image.fromarray(img.asnumpy(), mode='RGB')
            # # fig.save(save_name)

        print('load finished', len(data_list))
        return data_list, issame_list