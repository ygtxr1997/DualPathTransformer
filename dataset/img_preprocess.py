from PIL import Image
import numpy as np
import random
import copy
import os
from torchvision import transforms
import time
import math
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def BlockOcc(img, size, ratio):
    img_occ = copy.deepcopy(img)
    if ratio > 0:
        block_size = int((ratio * size * size) ** 0.5)
        occ = Image.fromarray(np.zeros([block_size, block_size], dtype=np.uint8))
        # occ = Image.fromarray(np.random.randn(block_size, block_size))
        randx = random.randint(0, size - block_size)
        randy = random.randint(0, size - block_size)
        img_occ.paste(occ, (randx, randy))
    return img_occ


class Randomblock(object):
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, img):
        ratio = random.randint(self.lo, self.hi) * 0.01
        img = BlockOcc(img, img.size[1], ratio)
        return img


class MskApply(object):
    def __call__(self, img, msk):
        img_occ = copy.deepcopy(img)
        img_occ = np.array(img_occ)
        msk = np.array(msk)
        msk[msk <= 128] = 0
        msk[msk > 128] = 1
        img_occ *= msk
        img_occ = Image.fromarray(img_occ)
        return img_occ


def RecOcc(img, size, ratio):
    img_occ = copy.deepcopy(img)
    if ratio > 0:
        block_size = int(size * size * ratio)

        width = random.randint(int(block_size / size) + 1, size - 1)
        height = int(block_size / width)
        randx = random.randint(0, size - width)
        randy = random.randint(0, size - height)

        img_occ = np.array(img_occ, dtype=np.uint8)
        img_occ[randy:randy + height, randx:randx + width] = random.randint(0, 255)

        img_occ = Image.fromarray(img_occ)
        msk = np.ones([size, size], np.uint8) * 255
        msk[randy:randy + height, randx:randx + width] = 0
        msk = Image.fromarray(msk)
    return img_occ, msk


class BaseTrans(object):
    def __call__(self, img):
        img = copy.deepcopy(img)
        random_flip = transforms.RandomHorizontalFlip()
        img = random_flip(img)
        msk = np.ones([img.size[1], img.size[0]], dtype=np.uint8) * 255
        msk = Image.fromarray(msk)
        return img, msk


class RandomRecOcc(object):
    def __init__(self, ):
        self.ratio = random.randint(1, 35) * 0.01

    def __call__(self, img):
        img = copy.deepcopy(img)
        img, msk = self._RecOcc(img, img.size[1])
        return img, msk

    def _RecOcc(self, img, size):
        # img_occ = copy.deepcopy(img)
        img_occ = img

        block_size = int(size * size * self.ratio)

        self.width = random.randint(int(block_size / size) + 1, size - 1)
        self.height = int(block_size / self.width)
        self.randx = random.randint(0, size - self.width)
        self.randy = random.randint(0, size - self.height)

        img_occ = np.array(img_occ, dtype=np.uint8)
        if len(img_occ.shape) == 2:  # gray
            self.gray_val = random.randint(0, 255)
            img_occ[self.randy:self.randy + self.height,
                    self.randx:self.randx + self.width] = self.gray_val
        elif len(img_occ.shape) == 3:  # rgb
            for c in range(3):
                self.rgb_val = random.randint(0, 255)
                img_occ[self.randy:self.randy + self.height,
                        self.randx:self.randx + self.width, c] = self.rgb_val

        img_occ = Image.fromarray(img_occ)

        msk = np.ones([size, size], np.uint8) * 255
        msk[self.randy:self.randy + self.height,
            self.randx:self.randx + self.width] = 0
        msk = Image.fromarray(msk)

        return img_occ, msk


class RandomConnectedPolygon(object):
    def __init__(self, connected_num=1, ratio=0.4, rand_gray_val=True, edge_num=7):
        self.connected_num = connected_num
        self.ratio = ratio
        self.rand_gray_val = rand_gray_val
        self.edge_num = edge_num

    def __call__(self, img):
        img = copy.deepcopy(img)
        face_arr = np.array(img)
        height, width = img.size[1], img.size[0]

        channel = 1 if len(face_arr.shape) == 2 else 3

        polygon = self._get_polygon(height, width)

        color_list = np.array([0, 0, 0])
        for c in range(channel):
            color_list[c] = random.randint(1, 255) if self.rand_gray_val else 255
        face_arr[polygon != 0] = color_list if channel == 3 else color_list[0]

        msk = np.ones([height, width], dtype=np.uint8) * 255
        msk[polygon != 0] = 0

        img_surpass = Image.fromarray(face_arr)
        msk = Image.fromarray(msk)

        return img_surpass, msk

    def _get_polygon(self, height, width):
        """
        Create random polygon. \n
        0: no occlusion \n
        1~255: occluded by random gray value \n
        """
        # start_time = time.time()
        polygon = np.zeros([height, width], dtype=np.uint8)

        point_cnt = (random.randint(4, 10))
        points = np.zeros((point_cnt, 2))

        center_x = random.randint(height // 5, 4 * height // 5)
        center_y = random.randint(width // 5, 4 * width // 5)

        big_radius = random.randint(height // 5, 1.3 * height // 5)
        small_radius = big_radius / random.uniform(1.3, 2.6)
        big_angle, small_angle = 0, 0  # [0, 2pi]

        # First point
        points[0] = self._calc_from_circle(big_radius, big_angle, center_x, center_y)

        for ind in range(1, point_cnt):
            big_angle += 2 * math.pi / point_cnt * random.uniform(0.7, 1.3)
            points[ind] = self._calc_from_circle(big_radius, big_angle, center_x, center_y)

            if random.random() > 0.5:
                small_angle += 2 * math.pi / point_cnt * random.uniform(0.6, 1.4)
                points[ind] = self._calc_from_circle(small_radius, small_angle, center_x, center_y)

        gray_val = random.randint(1, 255) if self.rand_gray_val else 255

        points = np.array([points], dtype=np.int32)
        cv2.fillPoly(polygon, points, gray_val)

        return polygon

    def _calc_from_circle(self, radius, angle, center_x, center_y):
        target_x = center_x + radius * math.cos(angle)
        target_y = center_y + radius * math.sin(angle)
        return np.array([int(target_x), int(target_y)])


class RandomConnectedOval(object):
    def __init__(self, connected_num=1, ratio=0.3, rand_gray_val=True):
        """
        Random occlude the origin face image by oval. \n
        :param connected_num: the num of connected blocks
        :param ratio: the sum pixels of cross line
        :param rand_gray_val: enable/unable the random gray value
        """
        self.connected_num = connected_num
        self.ratio = ratio
        self.rand_gray_val = rand_gray_val

    def __call__(self, img):
        img = copy.deepcopy(img)
        face_arr = np.array(img)
        height, width = img.size[1], img.size[0]

        channel = 1 if len(face_arr.shape) == 2 else 3

        oval = self._get_oval(height, width)

        color_list = np.array([0, 0, 0])
        for c in range(channel):
            color_list[c] = random.randint(1, 255) if self.rand_gray_val else 255
        face_arr[oval != 0] = color_list if channel == 3 else color_list[0]

        msk = np.ones([height, width], dtype=np.uint8) * 255
        msk[oval != 0] = 0

        img_surpass = Image.fromarray(face_arr)
        msk = Image.fromarray(msk)

        return img_surpass, msk

    def _get_oval(self, height, width):
        """
        Create random oval shape. \n
        0: no occlusion \n
        1~255: occluded by random gray value \n
        """
        oval = np.zeros([height, width], dtype=np.uint8)

        ch = random.randint(height // 5, 4 * height // 5)
        cw = random.randint(width // 5, 4 * width // 5)
        ah = random.randint(20, min(ch, height - ch))
        aw = int(height * width * self.ratio / (3.14 * ah))
        angle = 0
        gray_val = random.randint(1, 255) if self.rand_gray_val else 255

        cv2.ellipse(oval, (cw, ch), (aw, ah), angle, 0, 360, gray_val, -1)

        return oval


""" Random Glasses Occlusion
This transform is only used for training.
Init Params:
    - glasses_path: the path to glasses image folder
    - occ_height: we should resize the glasses images into the same height (default: 40)
    - occ_width: we should resize the glasses images into the same width (default: 89)
    - height_scale: the resized images can be randomly rescaled by -h_s to +h_s (h_s >= 1.0, default: 1.1)
    - width_scale: the resized images can be randomly rescaled by -w_s to +w_s (w_s >= 1.0, default: 1.1)
"""
class RandomGlasses(object):
    def __init__(self,
                 glasses_path: str = 'occluder/glasses_crop/',
                 occ_height: int = 40,
                 occ_width: int = 80,
                 height_scale: float = 1.1,
                 width_scale: float = 1.1,
                 ):
        self.glasses_root = glasses_path
        self.glasses_list = np.array(os.listdir(glasses_path))
        self.glasses_num = len(self.glasses_list)

        self.occ_height = occ_height
        self.occ_width = occ_width
        self.height_scale = height_scale
        self.width_scale = width_scale

        # Preload the image folder
        self.object_imgs = np.zeros((self.glasses_num,
                                     occ_height, occ_width, 4), dtype=np.uint8)  # (num, height, width, RGBA)
        for idx in range(self.glasses_num):
            object_path = os.path.join(self.glasses_root, self.glasses_list[idx])
            object = Image.open(object_path).convert('RGBA')  # [w, h]: (125, 40+)
            object = object.resize((occ_width, occ_height))
            self.object_imgs[idx] = np.array(object, dtype=np.uint8)  # [h, w, c=4]

    def __call__(self, img):
        mode = img.mode  # 'L' or 'RGB'
        height, width = img.size[1], img.size[0]
        occ_height = height * (self.occ_height / 120)
        occ_width = width * (self.occ_width / 120)

        """ 1. Get an occlusion image from the preloaded list, and resize it randomly """
        glasses = self.object_imgs[np.random.randint(0, self.glasses_num)]  # np-(h, w, RGBA)
        glasses = Image.fromarray(glasses, mode='RGBA')  # PIL-(h, w, RGBA)
        occ_width = int(occ_width * np.random.uniform(1 / self.width_scale, self.width_scale))  # w'
        occ_height = int(occ_height * np.random.uniform(1 / self.height_scale, self.height_scale))  # h'
        glasses = glasses.resize((occ_width, occ_height))  # PIL-(h', w', RGBA)

        """ 2. Split Alpha channel and RGB channels, and convert RGB channels into img.mode """
        alpha = np.array(glasses)[:, :, -1].astype(np.uint8)  # np-(h', w', A)
        glasses = glasses.convert(mode)  # PIL-(h', w', mode)

        """ 3. Generate top-left point (x, y) of occlusion """
        x_offset = int((0.12 + np.random.randint(-5, 6) * 0.02) * width)
        y_offset = int((0.3 + np.random.randint(-5, 6) * 0.01) * height)

        """ 4. Surpass the face by occlusion, based on np.array """
        face_arr = np.array(img)  # (H, W, mode)
        glasses_arr = np.array(glasses)  # (h', w', mode)

        face_crop = face_arr[y_offset: y_offset + occ_height,
                             x_offset: x_offset + occ_width]  # Crop the face according to the glasses position
        glasses_arr[alpha <= 10] = face_crop[alpha <= 10]  # 'Alpha == 0' denotes transparent pixel
        face_arr[y_offset: y_offset + occ_height,
                 x_offset: x_offset + occ_width] = glasses_arr  # Overlap, np-(H, W, mode)

        """ 5. Get occluded face and occlusion mask """
        img_glassesed = Image.fromarray(face_arr)  # PIL-(H, W, mode)

        msk_shape = (height, width) if mode == 'L' else (height, width, 3)
        msk = np.ones(msk_shape, dtype=np.uint8) * 255
        glasses_arr[alpha != 0] = 0  # occluded
        glasses_arr[alpha == 0] = 255  # clean
        msk[y_offset: y_offset + occ_height,
            x_offset: x_offset + occ_width] = glasses_arr
        msk = Image.fromarray(msk).convert('L')

        return img_glassesed, msk


class RandomGlassesList(object):
    def __init__(self,
                 glasses_path_list: list,
                 ):
        self.trans_list = []
        for glasses_path in glasses_path_list:
            self.trans_list.append(RandomGlasses(glasses_path))

    def __call__(self, img):
        img = copy.deepcopy(img)
        trans_idx = np.random.randint(0, len(self.trans_list))
        img_glassesed, msk = self.trans_list[trans_idx](img)
        return img_glassesed, msk


class Scarf(object):
    def __init__(self, scarf_path='in/occluder/scarf_crop/'):
        self.scarf_root = scarf_path
        self.scarf_list = np.array(os.listdir(scarf_path))
        self.scarf_num = len(self.scarf_list)

        self.object_imgs = np.zeros((self.scarf_num, 112, 112, 4))
        for idx in range(self.scarf_num):
            object_path = os.path.join(self.scarf_root, self.scarf_list[idx])
            object = Image.open(object_path).convert('RGBA')
            object = object.resize((112, 112))
            self.object_imgs[idx] = np.array(object)

    def __call__(self, img):
        img = copy.deepcopy(img)
        # scarf_path = os.path.join(self.scarf_root, self.scarf_list[random.randint(0, self.scarf_num - 1)])
        # scarf = Image.open(scarf_path).convert('RGBA')
        scarf = self.object_imgs[random.randint(0, self.scarf_num - 1)]
        scarf = Image.fromarray(scarf.astype('uint8')).convert('RGBA')

        alpha = np.array(scarf)[:, :, -1]  # channel A
        channel = 1 if len(img.split()) == 2 else 3
        mode = 'L' if channel == 1 else 'RGB'
        scarf = scarf.convert(mode)

        x_offset = int((0.02 + random.randint(-5, 5) * 0.001) * img.size[0])
        y_offset = int((0.6 + random.randint(-5, 5) * 0.01) * img.size[0])

        face_arr = np.array(img)
        scarf_arr = np.array(scarf)

        scarf_arr = scarf_arr[:min(scarf.size[1], img.size[1] - y_offset), :min(scarf.size[0], img.size[0] - x_offset)]
        alpha = alpha[:min(scarf.size[1], img.size[1] - y_offset), :min(scarf.size[0], img.size[0] - x_offset)]

        face_crop = face_arr[y_offset:y_offset + scarf_arr.shape[0], x_offset:x_offset + scarf_arr.shape[1]]
        face_crop[alpha != 0] = scarf_arr[alpha != 0]
        face_arr[y_offset:y_offset + scarf_arr.shape[0], x_offset:x_offset + scarf_arr.shape[1]] = face_crop

        img_scarfed = Image.fromarray(face_arr)

        msk_shape = (img.size[1], img.size[0]) if channel == 1 else (img.size[1], img.size[0], 3)
        msk = np.ones(msk_shape, dtype=np.uint8) * 255
        scarf_arr[alpha != 0] = 0  # occluded
        scarf_arr[alpha == 0] = 255  # clean
        msk[y_offset:y_offset + scarf_arr.shape[0], x_offset:x_offset + scarf_arr.shape[1]] = scarf_arr
        msk = Image.fromarray(msk).convert('L')
        return img_scarfed, msk


class RandomTrueObject(object):
    def __init__(self, object_path='/home/yuange/code/SelfServer/dongjiayu-light/in/occluder/object'):
        self.object_root = object_path
        self.object_list = np.array(os.listdir(object_path))
        self.object_num = len(self.object_list)

        self.object_imgs = np.zeros((self.object_num, 112, 112, 4))
        for idx in range(self.object_num):
            object_path = os.path.join(self.object_root, self.object_list[idx])
            object = Image.open(object_path).convert('RGBA')
            object = object.resize((112, 112))
            self.object_imgs[idx] = np.array(object)

    def __call__(self, img):
        # object_path = os.path.join(self.object_root, self.object_list[random.randint(0, self.object_num - 1)])
        # object = Image.open(object_path).convert('RGBA')
        img = copy.deepcopy(img)
        object = self.object_imgs[random.randint(0, self.object_num - 1)]
        object = Image.fromarray(object.astype('uint8')).convert('RGBA')
        scale_ratio = (random.randint(40, 50) / 100) * img.size[0] / object.size[0]
        object = object.resize((int(scale_ratio * object.size[0]),
                              int(scale_ratio * object.size[1])))

        alpha = np.array(object)[:, :, -1]  # channel A
        channel = 1 if len(img.split()) == 2 else 3
        mode = 'L' if channel == 1 else 'RGB'
        object = object.convert(mode)

        x_offset = int((random.randint(15, 70) * 0.01) * img.size[0])
        y_offset = int((random.randint(15, 70) * 0.01) * img.size[0])

        face_arr = np.array(img)
        object_arr = np.array(object)

        object_arr = object_arr[:min(object.size[1], img.size[1] - y_offset), :min(object.size[0], img.size[0] - x_offset)]
        alpha = alpha[:min(object.size[1], img.size[1] - y_offset), :min(object.size[0], img.size[0] - x_offset)]

        face_crop = face_arr[y_offset:y_offset + object_arr.shape[0], x_offset:x_offset + object_arr.shape[1]]
        face_crop[alpha != 0] = object_arr[alpha != 0]
        face_arr[y_offset:y_offset + object_arr.shape[0], x_offset:x_offset + object_arr.shape[1]] = face_crop

        img_objected = Image.fromarray(face_arr)

        msk_shape = (img.size[1], img.size[0]) if channel == 1 else (img.size[1], img.size[0], 3)
        msk = np.ones(msk_shape, dtype=np.uint8) * 255
        object_arr[alpha != 0] = 0  # occluded
        object_arr[alpha == 0] = 255  # clean
        msk[y_offset:y_offset + object_arr.shape[0], x_offset:x_offset + object_arr.shape[1]] = object_arr
        msk = Image.fromarray(msk).convert('L')

        return img_objected, msk


class RandomConnectedCrossLine(object):
    # TODO: Bug exists
    def __init__(self, connected_num=1, ratio=0.5, rand_gray_val=True):
        self.connected_num = connected_num
        self.ratio = ratio
        self.rand_gray_val = rand_gray_val

    def __call__(self, img):
        face_arr = np.array(img)
        height, width = img.size[1], img.size[0]

        channel = 1 if len(face_arr.shape) == 2 else 3

        cross_line = self._bfs(height, width)

        color_list = [0, 0, 0]
        for c in range(channel):
            color_list[c] = random.randint(1, 255) if self.rand_gray_val else 255
        face_arr[cross_line != 0] = color_list if channel == 3 else color_list[0]

        img_surpass = Image.fromarray(face_arr)
        cross_line = Image.fromarray(cross_line).convert('L')
        return img_surpass, cross_line

    def _bfs(self, height, width):
        """BFS to get connected graph. \n"""
        cross_line = np.zeros([height, width], dtype=np.uint8)
        px, py = random.randint(0, height - 1), random.randint(0, width - 1)
        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        gray_val = random.randint(1, 255) if self.rand_gray_val else 255

        q = []
        cnt, tot = 1, height * width * self.ratio
        q.append([px, py])
        cross_line[px][py] = gray_val
        while len(q) and cnt < tot:
            px, py = q.pop(0)
            for op in range(4):
                px, py = px + direction[op][0], py + direction[op][1]
                print(px, py)
                if 0 <= px < height and 0 <= py < width \
                        and cross_line[px][py] == 0 \
                        and random.random() > 0.1:
                    q.append([px, py])
                    cross_line[px][py] = gray_val
                    cnt += 1

        return cross_line


if __name__ == '__main__':
    img_in = Image.fromarray(np.ones((112, 112, 3), dtype=np.uint8) * 255)
    trans = RandomConnectedCrossLine()
    for i in range(1):
        img_out, msk = trans(img_in)
    print('img_out:', img_out.size, 'msk:', msk.size)
    print('img_out channel:', len(img_out.split()))
    img_out.save('img_preprocess.jpg')
    msk.save('img_preprocess_msk.jpg')