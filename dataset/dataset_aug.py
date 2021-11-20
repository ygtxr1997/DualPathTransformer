import os
import random
import numbers

import mxnet as mx
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms

from dataset.img_preprocess import RandomRecOcc, Glasses, Scarf, BaseTrans
from dataset.img_preprocess import RandomConnectedOval, RandomConnectedPolygon, RandomTrueObject


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FaceRandOccMask(data.Dataset):
    # read mxnet dataset
    def __init__(self, root_dir, local_rank,
                 img_transform=None, msk_transform=None,
                 is_train=True,
                 out_size=128, gray=True, norm=False):

        super(FaceRandOccMask, self).__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        path_mask_out_rec = os.path.join(root_dir, 'mask_out.rec')
        path_mask_out_idx = os.path.join(root_dir, 'mask_out.idx')
        path_mask_rec = os.path.join(root_dir, 'mask.rec')
        path_mask_idx = os.path.join(root_dir, 'mask.idx')

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.mask_out_rec = mx.recordio.MXIndexedRecordIO(path_mask_out_idx, path_mask_out_rec, 'r')
        self.mask_rec = mx.recordio.MXIndexedRecordIO(path_mask_idx, path_mask_rec, 'r')

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = np.array((int(header.label[0]), int(header.label[1])))
            self.imgidx = np.array(range(1, int(header.label[0])))  # mask_out.rec starts from index 1
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        self.img_tf = img_transform
        self.msk_tf = msk_transform
        self.output_size = out_size
        self.gray = gray
        self.norm = norm

        self.trans_occ = np.array([BaseTrans(), RandomRecOcc(),
                                  RandomConnectedOval(), RandomConnectedPolygon(),
                                  RandomTrueObject('./resources/occ/object'),
                                  Glasses('./resources/occ/glasses_crop/'),
                                  Scarf('./resources/occ/scarf_crop/')])
        self.is_train = is_train

        self.avg_time1 = AverageMeter()
        self.avg_time2 = AverageMeter()
        self.avg_time3 = AverageMeter()
        self.avg_time4 = AverageMeter()

    def __getitem__(self, index):

        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)

        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)

        if random.randint(1, 10) >= 9:
            mask_flag = True  # TODO: Close masked face
        else:
            mask_flag = False

        out_size = self.output_size
        if not mask_flag:
            img = mx.image.imdecode(img, to_rgb=1).asnumpy()  # original face, (112, 112, 3)

            img = Image.fromarray(img)

            # msk = Image.fromarray(np.ones((out_size, out_size), dtype=np.uint8) * 255)

            random_trans = self.trans_occ[random.randint(0, 6)]
            img, msk = random_trans(img)
        else:
            s1 = self.mask_out_rec.read_idx(idx)
            header, image = mx.recordio.unpack(s1)
            img_masked = mx.image.imdecode(image, to_rgb=1).asnumpy()  # masked face, rgb

            s2 = self.mask_rec.read_idx(idx)
            header, image = mx.recordio.unpack(s2)
            mask = mx.image.imdecode(image, to_rgb=1).asnumpy()  # mask, rgb

            img = Image.fromarray(img_masked)
            msk = Image.fromarray(mask).convert('L')

        # 1. Center Crop
        trans = transforms.Resize(out_size)
        img = trans(img)
        msk = trans(msk)

        # 2. Render Mask
        img_trans = img

        # 3. Random Crop - not used
        # crop_size = 112
        # left_top = (random.randint(0, 144 - crop_size - 1), random.randint(0, 144 - crop_size - 1))
        # img_trans = img_trans.crop((left_top[0], left_top[1], left_top[0] + crop_size, left_top[1] + crop_size))
        # msk = msk.crop((left_top[0], left_top[1], left_top[0] + crop_size, left_top[1] + crop_size))

        # 4. Random Horizontal Filp
        if random.randint(0, 10) >= 5:
            img_trans = img_trans.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        # 5. To Tensor
        if self.img_tf is not None:
            img_trans = self.img_tf(img_trans)

            # 5.1 Add Light to Masked Image
            light = self._get_Gauss(0, 0, out_size, out_size,
                                    center_x=np.array([1, 56, 111]),
                                    center_y=np.array([1, 56, 111]),
                                    R=224)
            scale = random.uniform(0.7, 1.4)
            light = light.astype(np.float32) * scale
            img_trans = img_trans * light

            # 5.2 gauss noise
            # if random.randint(0, 10) >= 5:
            #     gauss = 0.1 * np.random.normal(loc=0, scale=1, size=(out_size, out_size))
            #     img_trans = img_trans + gauss.astype(np.float32)

            img_trans[img_trans > 1.0] = 1.0
            img_trans[img_trans < 0.0] = 0.0

        if self.msk_tf is not None:
            if mask_flag:
                # binarization
                msk = np.array(msk)
                msk_tmp = np.ones((out_size, out_size)) * 255  # white image
                msk_tmp[msk <= 128] = 0  # paint mask

                # add random lightness to mask
                msk_light = np.zeros((3, out_size, out_size), dtype=np.float32)  # offset of mask
                msk_light[0] = (msk_tmp // 128 * (-1) + 1).astype(np.float32)  # 1:mask, 0:face
                msk_light[1] = (msk_tmp // 128 * (-1) + 1).astype(np.float32)  # 1:mask, 0:face
                msk_light[2] = (msk_tmp // 128 * (-1) + 1).astype(np.float32)  # 1:mask, 0:face

                rescale_map = np.zeros((out_size, out_size), dtype=np.float32)

                left_top_x, left_top_y = 1, 40 + random.randint(-20, 20)
                right_down_x, right_down_y = 111, 100 + random.randint(-20, 10)

                trans_type = random.randint(0, 10)
                if trans_type >= 7:
                    Gauss_map = self._get_Gauss(left_top_x, left_top_y, right_down_x, right_down_y,
                                                center_x=np.array([1, 56, 111]),
                                                center_y=np.array([1, 56, 111]))
                    Gauss_map = (Gauss_map - 0.5) * 2 * 0.4 * (random.randint(0, 1) * 2 - 1)

                    rescale_map[left_top_x:right_down_x,
                                left_top_y:right_down_y] = np.transpose(Gauss_map)  # min:-0.4, max:0.4
                elif trans_type >= 5:
                    rescale_map[left_top_x:right_down_x,
                                left_top_y:right_down_y] = np.random.randn(right_down_x - left_top_x,
                                                                           right_down_y - left_top_y)
                else:
                    left_top_x, right_down_x = 40 + random.randint(-20, 20), 100 + random.randint(-20, 10)
                    block_map = np.zeros((out_size, out_size), dtype=np.float32)
                    block_map[left_top_x:right_down_x,
                              left_top_y:right_down_y] = random.randint(0, 1) * 2 - 1
                    msk_light = msk_light * block_map

                if trans_type >= 5:
                    for c in range(3):
                        if random.randint(0, 10) >= 5:
                            msk_light[c] = msk_light[c] * rescale_map
                        else:
                            msk_light[c] = 0

                if self.gray:
                    # msk_light_gray = np.zeros((1, out_size, out_size), dtype=np.float32)  # offset of mask
                    msk_light_gray = (0.2989 * msk_light[0] + 0.5870 * msk_light[1] + 0.1140 * msk_light[2]) / 3
                    msk_light = msk_light_gray

                img_trans = img_trans - msk_light
                img_trans[img_trans > 1.0] = 1.0
                img_trans[img_trans < 0.0] = 0.0

                msk = torch.from_numpy(msk_tmp // 255).long()
            else:
                msk = self.msk_tf(msk)

        # 6. Normalize
        if self.gray:
            norm = transforms.Normalize(mean=[0.5], std=[0.5])
        else:
            norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.norm:
            img_trans = norm(img_trans)

        return img_trans, msk, label

    def __len__(self):
        return len(self.imgidx)

    def _get_Gauss(self, left_top_x, left_top_y, right_down_x, right_down_y, center_x=None, center_y=None, R=-1):

        if center_x is None:
            center_x = np.zeros(3)
        if center_y is None:
            center_y = np.zeros(3)

        IMAGE_HEIGHT = right_down_y - left_top_y
        IMAGE_WIDTH = right_down_x - left_top_x

        if len(center_x) == 0 and len(center_y) == 0:
            center_x = left_top_x + (right_down_x - left_top_x) * random.random()
            center_y = left_top_y + (right_down_y - left_top_y) * random.random()
        else:
            center_x = center_x[random.randint(0, len(center_x) - 1)]
            center_y = center_y[random.randint(0, len(center_y) - 1)]

        if R < 0:
            R = max(IMAGE_WIDTH, IMAGE_HEIGHT)

        import numpy.matlib
        mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
        mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

        x1 = np.arange(IMAGE_WIDTH)
        x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

        y1 = np.arange(IMAGE_HEIGHT)
        y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
        y_map = np.transpose(y_map)

        Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)
        Gauss_map = np.exp(-0.5 * Gauss_map / R)

        return Gauss_map  # min:0, max:1


class Msk2Tenser(object):
    def __call__(self, msk):
        msk = np.array(msk, dtype=np.int32)
        msk[msk != 255] = 0
        msk[msk == 255] = 1
        msk = torch.from_numpy(msk).int()

        return msk