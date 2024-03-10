from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from torchvision.transforms import InterpolationMode
import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:  # 打开一个二进制文件
        with Image.open(f) as img:
            return img.convert('RGB')
    # 将二进制文件 如.npy文件转换为图片RGB格式


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 # img_ext='.jpg'):
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        # self.interp = Image.ANTIALIAS # 原
        self.interp = InterpolationMode.NEAREST
        # Image.ANTIALIAS 是 Pillow 中的一个插值方法常量。
        # ANTIALIAS 表示抗锯齿插值方法，它通常用于在图像缩放时平滑处理图像，以减少锯齿效应和像素化。

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            # 这段代码的目的是为数据增强操作设置随机参数，以便在训练过程中对输入图像进行随机的亮度、对比度、饱和度和色调的变换，以提高模型的鲁棒性。
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        # self.resize的尺度也是一个字典。
        for i in range(self.num_scales):
            s = 2 ** i
            # s = 2 ** i：这一行计算了当前尺度 i 对应的缩放因子 s。
            # 2 ** i 表示 2 的 i 次方，它用于确定缩放比例。
            # 在每次迭代中，s 的值都会不同，表示不同的尺度。
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)  # interpolation是插值方法。
        # 主要目的是为了在不同尺度下处理输入图像，以增加模型的多尺度感知能力

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        # 预处理
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for k in list(inputs):
            # 这里是将inputs的键值变成列表进行遍历。
            frame = inputs[k]
            if "color" in k:
                # 如果是对应的图像，即（”color“，-1/0/1，-1）这个键.
                n, im, i = k

                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            # 这个i感觉可以理解为对应的缩放尺度啊！
            # self.resize 是一个字典，其中，key是对应要缩放的大小，value是一个transform方法，用于缩放。

        for k in list(inputs):
            f = inputs[k]

            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            # color_aug 是一个transforms.ColorJitter的方法，用于调整图像的对比度、亮度等。
        # 图像增强
        # ######################## 至此预处理结束 ###################

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
            （视频序列的方式 ，一共三帧）
        or
            "s" for the opposite image in the stereo pair.
            （双目图像对的方式）

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
            (下采样的比例，2的几次幂)
        """
        inputs = {}

        # 训练时进行数据增强、翻转的判断。
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        # filenames是一个包含test_file.txt（eigen划分的用于测试的图片的信息）的列表。
        # 原txt文件中，每行有三个数据，首先是文件名（日期） 然后是具体的图片 最后是视图。
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
            # 帧的索引，即具体图片。
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
            # 哪一边的视图。
        else:
            side = None

        for i in self.frame_idxs:
            # frame_idxs只有两种：（-1，0，1）用于视频序列；（s）用于双目图像对。

            if i == "s":
                # 双目图像对的训练方式。
                other_side = {"r": "l", "l": "r"}[side]
                # 找到另一边的视图。
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                # 视频序列训练方式。
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # 调整相机参数K来对应缩放的部分
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)
            # K的逆矩阵

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 有50%的几率继续颜色的增强
        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # 将以上得到的inputs字典进行预处理。
        self.preprocess(inputs, color_aug)
        # 处理完后，又将inputs这个字典扩大了。

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        # 删除一些冗余的内容。

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            # 双目图像对训练。
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    # 以下三者都是抽象函数，需要在派生类中实现。
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
