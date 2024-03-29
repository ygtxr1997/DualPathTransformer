import os
import copy

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import imutils
from tqdm import tqdm

from dataset.RealOcc.utils.utils import get_srcNmask
from dataset.RealOcc.utils.utils import get_randomOccluderNmask, get_occluderNmask
from dataset.RealOcc.utils.utils import get_occluder_augmentor, get_src_augmentor
from dataset.RealOcc.utils.utils import RandomOccluderNmask
from dataset.RealOcc.utils.utils import OccluderNmask

from dataset.RealOcc.utils.utils import augment_occluder
from dataset.RealOcc.utils.utils import angle3pt
# from dataset.RealOcc.utils import colour_transfer
from dataset.RealOcc.utils.paste_over import paste_over
from dataset.RealOcc.utils import random_shape_generator

real_occ_path = {
    '11k-hands-img': '/tmp/train_tmp/real_occ/11k-hands_img',
    '11k-hands-msk': '/tmp/train_tmp/real_occ/11k-hands_masks',
    '11k-hands-txt': '/tmp/train_tmp/real_occ/11k_hands_sample.txt',
    'coco-img': '/tmp/train_tmp/real_occ/object_image_sr',
    'coco-msk': '/tmp/train_tmp/real_occ/object_mask_x4',
    'dtd': '/tmp/train_tmp/real_occ/dtd/images',
}

""" RealOcc (CVPRW'22)
This transform is only used for training.
Init Params:
    - occ_type: occlusion type
"""
class RealOcc(object):
    def __init__(self,
                 occ_type: str = 'hand',
                 split: str = 'train',
                 ):
        self.occ_type = occ_type

        self.on = None
        self.rom = None
        self.split = split
        if occ_type == 'hand':
            sample_path = real_occ_path['11k-hands-txt']
            img_path = real_occ_path['11k-hands-img']
            mask_path = real_occ_path['11k-hands-msk']
            occluders_list = get_occluders_list_from_txt(sample_path)
            occluders_list = self._split_samples(occluders_list, split)
            self.on = OccluderNmask(occluders_list=occluders_list,
                                    img_path=img_path,
                                    mask_path=mask_path)
        elif occ_type == 'coco':
            img_path = real_occ_path['coco-img']
            mask_path = real_occ_path['coco-msk']
            occluders_list = get_occluders_list_from_path(img_path)
            occluders_list = self._split_samples(occluders_list, split)
            self.on = OccluderNmask(occluders_list=occluders_list,
                                    img_path=img_path,
                                    mask_path=mask_path)
        elif occ_type == 'rand':
            img_path = real_occ_path['dtd']
            self.rom = RandomOccluderNmask(dtd_folder=img_path)
        else:
            raise KeyError('Occlusion type not supported.')

    def __call__(self, ori_img):
        ori_img = copy.deepcopy(ori_img)
        if self.occ_type == 'rand':
            occluder_img, occluder_mask = self.rom.get_img_mask()  # very slow
        else:
            occluder_img, occluder_mask = self.on.get_img_mask()

        randomOcclusion: bool = (self.occ_type == 'rand')

        w, h = ori_img.size
        src_img = np.array(ori_img)
        occluder_img = cv2.resize(occluder_img, ori_img.size)
        occluder_mask = cv2.resize(occluder_mask, ori_img.size)

        src_mask = np.ones((h, w), dtype=np.uint8)
        src_rect = cv2.boundingRect(src_mask)

        occluder_augmentor = get_occluder_augmentor()
        occluder_img, occluder_mask = augment_occluder(
            occluder_augmentor, occluder_img, occluder_mask, src_rect
        )
        occluder_coord = np.random.uniform([src_rect[0], src_rect[1]],
                                           [src_rect[0] + src_rect[2], src_rect[1] + src_rect[3]])

        src_center = (src_rect[0] + (src_rect[2] / 2), (src_rect[1] + src_rect[3] / 2))
        rotation = angle3pt((src_center[0], occluder_coord[1]), src_center, occluder_coord)
        if occluder_coord[1] > src_center[1]:
            rotation = rotation + 180
        occluder_img = imutils.rotate_bound(occluder_img, rotation)
        occluder_mask = imutils.rotate_bound(occluder_mask, rotation)
        occluder_img = self._resize_with_padding(occluder_img, ratio=0.75)
        occluder_mask = self._resize_with_padding(occluder_mask, ratio=0.75)

        # overlay occluder to src images
        occlusion_mask = np.zeros(src_mask.shape, np.uint8)
        occlusion_mask[(occlusion_mask > 0) & (occlusion_mask < 255)] = 255
        # paste occluder to src image
        result_img, result_mask, occlusion_mask = paste_over(occluder_img, occluder_mask, src_img, src_mask,
                                                             occluder_coord, occlusion_mask,
                                                             randomOcclusion)

        # augment occluded image
        image_augmentor = get_src_augmentor()
        transformed = image_augmentor(image=result_img, mask=result_mask, mask1=occlusion_mask)
        result_img, result_mask, occlusion_mask = transformed["image"], transformed["mask"], transformed["mask1"]
        result_img = Image.fromarray(result_img)
        occlusion_mask = 255 - occlusion_mask  # 0:occ, 255:face
        occlusion_mask = Image.fromarray(occlusion_mask)

        return result_img, occlusion_mask

    @staticmethod
    def _resize_with_padding(arr: np.ndarray, ratio: float):
        h, w = arr.shape[0], arr.shape[1]
        wr, hr = int(w * ratio), int(h * ratio)
        arr = cv2.resize(arr, (wr, hr))
        p = (w - wr) // 2
        arr = cv2.copyMakeBorder(arr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        arr = cv2.resize(arr, (w, h))
        return arr

    @staticmethod
    def _split_samples(sample_list: list, split: str, ratio: float = 0.8):
        n = len(sample_list)
        middle = int(n * ratio)
        if split == 'train':
            lo, hi = 0, middle
        else:
            lo, hi = middle, n
        return sample_list[lo: hi]


def get_occluders_list_from_txt(txt: str = '/gavin/datasets/msml/real_occ/11k_hands_sample.txt'):
    occluders_list = []
    with open(txt, 'r') as file:
        for line in file:
            line = line.strip('\n')
            occluders_list.append(line)
    return occluders_list


def get_occluders_list_from_path(path: str):
    occluders_list = os.listdir(path)
    return occluders_list


def add_occ(ori_img: Image,
            occ_type: str = 'hand',
            ) -> Image:
    occluders_list = []
    sample_path = None
    img_path = None
    mask_path = None

    if occ_type == 'hand':
        sample_path = real_occ_path['11k-hands-txt']
        img_path = real_occ_path['11k-hands-img']
        mask_path = real_occ_path['11k-hands-msk']
        occluders_list = get_occluders_list_from_txt(sample_path)
    elif occ_type == 'coco':
        img_path = real_occ_path['coco-img']
        mask_path = real_occ_path['coco-msk']
        occluders_list = get_occluders_list_from_path(img_path)
    elif occ_type == 'rand':
        img_path = real_occ_path['dtd']
    else:
        raise KeyError('Occlusion type not supported.')

    if occ_type == 'rand':
        from dataset.RealOcc.utils.utils import RandomOccluderNmask
        rom = RandomOccluderNmask(dtd_folder=img_path, mask_shape=112)
        occluder_img, occluder_mask = rom.get_img_mask()
    else:
        from dataset.RealOcc.utils.utils import OccluderNmask
        on = OccluderNmask(occluders_list=occluders_list,
                           img_path=img_path,
                           mask_path=mask_path)
        occluder_img, occluder_mask = on.get_img_mask()

    # args
    randomOcclusion: bool = (occ_type == 'rand')

    # src_img= cv2.imread(os.path.abspath(os.path.join(img_path,image_file)),-1)
    # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    w, h = ori_img.size
    src_img = np.array(ori_img)
    cv2.resize(occluder_img, ori_img.size)
    cv2.resize(occluder_mask, ori_img.size)

    # src_mask= cv2.imread(mask_path+f"{img_name}.png")
    # src_mask=cv2.resize(src_mask,(1024,1024),interpolation= cv2.INTER_LANCZOS4)
    # src_mask=cv2.cvtColor(src_mask,cv2.COLOR_RGB2GRAY)
    src_mask = np.ones((h, w), dtype=np.uint8)

    src_rect = cv2.boundingRect(src_mask)

    occluder_augmentor = get_occluder_augmentor()
    occluder_img, occluder_mask = augment_occluder(
        occluder_augmentor, occluder_img, occluder_mask, src_rect
    )
    occluder_coord = np.random.uniform([src_rect[0], src_rect[1]],
                                       [src_rect[0] + src_rect[2], src_rect[1] + src_rect[3]])

    src_center = (src_rect[0] + (src_rect[2] / 2), (src_rect[1] + src_rect[3] / 2))
    rotation = angle3pt((src_center[0], occluder_coord[1]), src_center, occluder_coord)
    if occluder_coord[1] > src_center[1]:
        rotation = rotation + 180
    occluder_img = imutils.rotate_bound(occluder_img, rotation)
    occluder_mask = imutils.rotate_bound(occluder_mask, rotation)

    # overlay occluder to src images
    occlusion_mask = np.zeros(src_mask.shape, np.uint8)
    occlusion_mask[(occlusion_mask > 0) & (occlusion_mask < 255)] = 255
    # paste occluder to src image
    result_img, result_mask, occlusion_mask = paste_over(occluder_img, occluder_mask, src_img, src_mask,
                                                         occluder_coord, occlusion_mask,
                                                         randomOcclusion)

    # augment occluded image
    image_augmentor = get_src_augmentor()
    transformed = image_augmentor(image=result_img, mask=result_mask, mask1=occlusion_mask)
    result_img, result_mask, occlusion_mask = transformed["image"], transformed["mask"], transformed["mask1"]
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    result_img = Image.fromarray(result_img)
    occlusion_mask = Image.fromarray(occlusion_mask)

    return result_img, occlusion_mask


class IJBDataset(Dataset):
    def __init__(self,
                 ijb_folder: str = '/gavin/datasets/msml/ijb/IJBB/loose_crop',
                 ):
        super(IJBDataset, self).__init__()
        self.root = ijb_folder
        self.img_list = os.listdir(ijb_folder)

        from datasets.augment.rand_occ import RandomRealObject, RandomGlasses
        self.occ_trans = RandomRealObject('/gavin/code/MSML/datasets/augment/occluder/object_test/')
        # self.occ_trans = RandomGlasses('/gavin/code/MSML/datasets/augment/occluder/eleglasses_crop/')

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = Image.open(os.path.join(self.root, img_name), 'r')
        img, occ = self.occ_trans(img)
        img = np.array(img)
        return img, img_name

    def __len__(self):
        return len(self.img_list)


def iterate_ijb(ijb_folder: str = '/gavin/datasets/msml/ijb/IJBB/loose_crop',
                out_folder: str = '/gavin/datasets/msml/ijb/IJBB/eyeglasses',
                ):
    print('output to %s' % out_folder)
    os.makedirs(out_folder, exist_ok=True)
    ijb_dataset = IJBDataset(ijb_folder=ijb_folder)
    eval_loader = DataLoader(dataset=ijb_dataset,
                             batch_size=1,
                             num_workers=24,
                             )

    idx = 0
    for batch in tqdm(eval_loader):
        idx += 1
        img, img_name = batch
        img = Image.fromarray(img[0].numpy())
        img_name = img_name[0]
        img.save(os.path.join(out_folder, img_name))
        if idx > 100:
            continue
            # exit(0)


if __name__ == '__main__':
    import time
    import random

    np.random.seed(4)
    random.seed(0)

    demo_inputs = ['1', '10', '32'] * 5

    ''' function format '''
    # for demo_input in demo_inputs:
    #     img = Image.open('demo/%s.jpg' % demo_input, 'r')
    #     res, occ = add_occ(img, occ_type='hand')
    #     res.save('demo/%s_occ.jpg' % demo_input)
    #     occ.save('demo/%s_msk.jpg' % demo_input)

    ''' offline occlusion to IJB '''
    # iterate_ijb(
    #     ijb_folder='/gavin/datasets/msml/ijb/IJBC/loose_crop',
    #     out_folder='/gavin/datasets/msml/ijb/IJBC/object',
    # )

    ''' class format '''
    ro = RealOcc(occ_type='coco')
    idx = 0
    for demo_input in demo_inputs:
        img = Image.open('demo/%s.jpg' % demo_input, 'r')
        img = img.resize((112, 112))
        start = time.time()
        res, occ = ro.__call__(img)
        print('cost time: %d ms' % int((time.time() - start) * 1000))
        res.save('demo/%s_%d_occ.jpg' % (demo_input, idx))
        occ.save('demo/%s_%d_msk.jpg' % (demo_input, idx))
        idx += 1
