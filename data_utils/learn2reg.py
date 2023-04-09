import os
import torch
from math import pi
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import itertools

from models.transformer import AffineTransformer
from data_utils.dir_lab import generate_affine_image, crop_center, normalize_numpy


class Learn2Reg(data.Dataset):

    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        if mode == 'train':
            self.file_index = list(range(1, 21))
        else:
            self.file_index = list(range(21, 31))

        self.file_list = self.get_img_pairs()
        # print(self.file_list.index((r'dataset/learn2reg\test\image\case28\case_028_insp_1.mhd', r'dataset/learn2reg\test\image\case28\case_028_insp_4.mhd')))
        # print(self.file_list)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # 读取预处理后的图片用于训练和测试
        image_pair = list(self.file_list[item])
        fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(image_pair[0]))[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(sitk.ReadImage(image_pair[1]))[np.newaxis, ...]
        # 读取对应的mask
        fixed_mask = image_pair[0].replace('image', 'mask')
        moving_mask = image_pair[1].replace('image', 'mask')

        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask))[np.newaxis, ...]
        moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask))[np.newaxis, ...]
        # 只返回包含肺部的区域
        return torch.from_numpy(fixed_img), torch.from_numpy(moving_img), \
               torch.from_numpy(fixed_mask), torch.from_numpy(moving_mask)

    def get_img_pairs(self):
        mode_root = os.path.join(self.root, self.mode, 'image')
        file_list = []
        for index in self.file_index:
            case_path = os.path.join(mode_root, f'case{index}')
            mhd_list = glob(case_path + '/*.mhd')
            file_list.extend(itertools.combinations(mhd_list, 2))
        return file_list


def resize_image_mask(img, mask, new_shape, spacing):
    assert img.shape == mask.shape
    shape = img.shape
    # print(spacing)
    # print(shape)
    ratio = np.asarray(shape)[::-1] / np.asarray(new_shape)
    # print(ratio)
    img = torch.from_numpy(np.expand_dims(img, axis=(0, 1)).copy()).float()
    mask = torch.from_numpy(np.expand_dims(mask, axis=(0, 1)).copy()).float()
    img = F.interpolate(img, size=new_shape, mode='trilinear', align_corners=True)
    mask = F.interpolate(mask, size=new_shape, mode='trilinear', align_corners=True)

    # 转换为numpy
    img = np.squeeze(np.asarray(img))
    mask = np.squeeze(np.asarray(mask))

    mask = np.where(mask > 0.2, 1, 0)
    mask = mask.astype(np.short)
    # 使用闭操作进行处理
    mask = sitk.GetImageFromArray(mask)
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(3)
    bm.SetForegroundValue(1)
    mask = bm.Execute(mask)
    mask = sitk.GetArrayFromImage(mask)

    return img.astype(np.float32), mask, spacing * ratio


def random_affine_train():
    img_root = r'H:\datasets\Learn2Reg2020'
    save_root = '../dataset/learn2reg/train'
    img_save_root = os.path.join(save_root, 'image')
    mask_save_root = os.path.join(save_root, 'mask')
    # 训练集路径
    train_root = os.path.join(img_root, 'training')

    aff_trans = AffineTransformer(ndim=3, coord_dim=1)
    # 遍历训练数据集中的吸气图像用于仿射变换
    for i in range(1, 21):
        img_save_path = os.path.join(img_save_root, f'case{i}')
        mask_save_path = os.path.join(mask_save_root, f'case{i}')
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        if i < 10:
            i = f'00{i}'
        else:
            i = f'0{i}'
        img_path = os.path.join(train_root, 'scans', f'case_{i}_insp.nii.gz')
        mask_path = img_path.replace('scans', 'lungMasks')

        # 读取数据
        sitk_img = sitk.ReadImage(img_path)
        sitk_mask = sitk.ReadImage(mask_path)
        # 读取源数据的信息
        spacing = sitk_img.GetSpacing()

        img = sitk.GetArrayFromImage(sitk_img)[::-1] - 1024
        img = np.clip(img, a_min=-1024, a_max=3071)
        mask = sitk.GetArrayFromImage(sitk_mask)[::-1]
        # TODO:resize图像以及mask至128*128*128，注意spacing的变换
        img, mask, spacing = resize_image_mask(img, mask, (128, 128, 128), spacing)

        # 对img和mask进行随机变换
        niter = 16
        shape = img.shape

        img = np.pad(img, pad_width=(100,), mode='constant', constant_values=(-2048,))
        mask = np.pad(mask, pad_width=(100,), mode='constant', constant_values=(0,))
        for j in range(niter):
            # 对原图以及肺掩码进行变换
            warped_img, warped_mask = generate_affine_image(img, mask, aff_trans)
            # 裁剪图像以及mask
            warped_img = crop_center(warped_img, *shape)
            warped_mask = crop_center(warped_mask, *shape)

            # TODO:normalize图像数据范围至[0,255]
            warped_img = normalize_numpy(warped_img)
            # 转换mask为二值图
            warped_mask = np.where(warped_mask > 0.4, 1, 0)
            warped_mask = warped_mask.astype(np.short)
            # 线性插值mask不是二值图，最邻近插值不够平滑
            # 使用闭操作进行处理
            warped_mask = sitk.GetImageFromArray(warped_mask)
            bm = sitk.BinaryMorphologicalClosingImageFilter()
            bm.SetKernelType(sitk.sitkBall)
            bm.SetKernelRadius(3)
            bm.SetForegroundValue(1)
            warped_mask = bm.Execute(warped_mask)
            warped_mask = sitk.GetArrayFromImage(warped_mask)
            # 设置img、mask保存路径
            img_save_file = os.path.join(img_save_path, f"case_{i}_insp_{j}.mhd")
            mask_save_file = os.path.join(mask_save_path, f"case_{i}_insp_{j}.mhd")
            print(img_save_file, mask_save_file)
            warped_img = sitk.GetImageFromArray(warped_img.astype(np.float32))
            warped_mask = sitk.GetImageFromArray(warped_mask)
            # 设置转换后图像的信息
            warped_img.SetSpacing(spacing)
            warped_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
            # warped_img.SetOrigin(origin)

            warped_mask.SetSpacing(spacing)
            warped_mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
            # warped_mask.SetOrigin(origin)
            # 保存转换后的.mhd文件
            sitk.WriteImage(warped_img, img_save_file)
            sitk.WriteImage(warped_mask, mask_save_file)


def random_affine_test():
    img_root = r'H:\datasets\Learn2Reg2020'
    save_root = '../dataset/learn2reg/test'
    img_save_root = os.path.join(save_root, 'image')
    mask_save_root = os.path.join(save_root, 'mask')
    # 测试集路径
    test_root = os.path.join(img_root, 'testData')

    aff_trans = AffineTransformer(ndim=3, coord_dim=1)
    # 遍历训练数据集中的吸气图像用于仿射变换
    for i in range(21, 31):
        img_save_path = os.path.join(img_save_root, f'case{i}')
        mask_save_path = os.path.join(mask_save_root, f'case{i}')
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        i = f'0{i}'
        img_path = os.path.join(test_root, 'scans', f'case_{i}_insp.nii.gz')
        mask_path = img_path.replace('scans', 'lungMasks')

        # 读取数据
        sitk_img = sitk.ReadImage(img_path)
        sitk_mask = sitk.ReadImage(mask_path)
        # 读取源数据的信息
        spacing = sitk_img.GetSpacing()
        origin = sitk_img.GetOrigin()

        img = sitk.GetArrayFromImage(sitk_img)[::-1] - 1024
        mask = sitk.GetArrayFromImage(sitk_mask)[::-1]
        # TODO:resize图像以及mask至128*128*128，注意spacing的变换
        img, mask, spacing = resize_image_mask(img, mask, (128, 128, 128), spacing)
        # 对img和mask进行随机变换
        niter = 8
        shape = img.shape

        img = np.pad(img, pad_width=(100,), mode='constant', constant_values=(-1024,))
        mask = np.pad(mask, pad_width=(100,), mode='constant', constant_values=(0,))
        for j in range(niter):
            # 对原图以及肺掩码进行变换
            warped_img, warped_mask = generate_affine_image(img, mask, aff_trans)
            # 裁剪图像以及mask
            warped_img = crop_center(warped_img, *shape)
            warped_mask = crop_center(warped_mask, *shape)
            # TODO:normalize图像数据范围至[0,255]
            warped_img = normalize_numpy(warped_img)
            # 转换为short类型
            warped_mask = np.where(warped_mask > 0.4, 1, 0)
            # warped_img = warped_img.astype(np.short)
            warped_mask = warped_mask.astype(np.short)
            # 线性插值mask不是二值图，最邻近插值不够平滑
            # 使用闭操作进行处理
            warped_mask = sitk.GetImageFromArray(warped_mask)
            bm = sitk.BinaryMorphologicalClosingImageFilter()
            bm.SetKernelType(sitk.sitkBall)
            bm.SetKernelRadius(3)
            bm.SetForegroundValue(1)
            warped_mask = bm.Execute(warped_mask)
            warped_mask = sitk.GetArrayFromImage(warped_mask)
            # 设置img、mask保存路径
            img_save_file = os.path.join(img_save_path, f"case_{i}_insp_{j}.mhd")
            mask_save_file = os.path.join(mask_save_path, f"case_{i}_insp_{j}.mhd")
            print(img_save_file, mask_save_file)
            warped_img = sitk.GetImageFromArray(warped_img.astype(np.float32))
            warped_mask = sitk.GetImageFromArray(warped_mask)
            # 设置转换后图像的信息
            warped_img.SetSpacing(spacing)
            warped_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
            # warped_img.SetOrigin(origin)

            warped_mask.SetSpacing(spacing)
            warped_mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
            # warped_mask.SetOrigin(origin)
            # 保存转换后的.mhd文件
            sitk.WriteImage(warped_img, img_save_file)
            sitk.WriteImage(warped_mask, mask_save_file)


if __name__ == '__main__':
    random_affine_test()
    test_data = Learn2Reg(root='../dataset/learn2reg',mode='test')
    print(len(test_data))
    fix, moving, fix_mask, moving_mask = test_data[0]
    print(moving_mask.shape)
    pass
