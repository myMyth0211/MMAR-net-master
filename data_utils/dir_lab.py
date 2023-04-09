import os
import torch
from math import pi
from torch.utils import data
from torchvision import transforms
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import itertools

from models.transformer import AffineTransformer


class DIR_Lab(data.Dataset):

    def __init__(self, img_root, mask_root, mode='train'):
        assert mode in ['train', 'test']
        self.img_root = img_root
        self.mask_root = mask_root
        self.mode = mode
        self.tranform = None

        self.file_path = os.path.join(img_root, mode)
        self.image_pair_list = self.get_image_pair()
        if mode=='train':
            self.image_pair_list=self.image_pair_list[::-1]

    def __len__(self):
        return len(self.image_pair_list)

    def __getitem__(self, item):
        # 读取预处理后的图片用于训练和测试
        image_pair = list(self.image_pair_list[item])
        fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(image_pair[0]))[np.newaxis, ...]
        moving_img = sitk.GetArrayFromImage(sitk.ReadImage(image_pair[1]))[np.newaxis, ...]
        # 读取对应的mask
        fixed_mask = image_pair[0].replace(self.img_root, self.mask_root)
        moving_mask = image_pair[1].replace(self.img_root, self.mask_root)

        fixed_mask = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask))[np.newaxis, ...]
        moving_mask = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask))[np.newaxis, ...]
        # 只返回包含肺部的区域
        return torch.from_numpy(fixed_img), torch.from_numpy(moving_img), \
               torch.from_numpy(fixed_mask), torch.from_numpy(moving_mask)

    def get_image_pair(self):
        image_pair_list = []
        for case in os.listdir(self.file_path):
            # 遍历case0-case10
            case_path = os.path.join(self.file_path, case)
            img_list = glob(case_path + '/*.mhd')
            # 将每个case中的所有CT图像计算组合数
            image_pair_list.extend(itertools.combinations(img_list, 2))
        return image_pair_list


def normalize_numpy(image_array, min_value=-1024, max_value=3071):
    image_array = np.clip(image_array, a_min=min_value, a_max=max_value)
    image_array = (image_array - min_value) / (max_value - min_value) * 255
    return image_array.astype(np.float)


def resize_numpy(image, spacing, value=0, shape=(128, 128, 128)):
    image_shape = image.shape
    step = int(image_shape[-1] // shape[-1])
    image = image[:, ::step, ::step]  # 重采样以缩小xy轴上的长度，注意size应该能整除128
    # 更新采样后的spacing
    spacing[0] = spacing[0] * step
    spacing[1] = spacing[1] * step
    diff = image_shape[0] - shape[0]
    padding = [diff // 2, diff - diff // 2]
    if diff > 0:
        image = image[padding[0]:-padding[1]]
    elif diff < 0:
        image = np.pad(image, ((abs(padding[0]), abs(padding[1])), (0, 0), (0, 0)),
                       'constant', constant_values=value)
    return image, spacing


def resize_and_normalize_image(modes):
    """
    To resize and normalize the image of Dir-Lab to (128,128,128) and float (0.0,255.0)
    To save the pre-processed image
    """
    img_save_root = '4DCT_pro'
    mask_save_root = 'mask_pro'
    dir_root = '../aff_4DCT'

    # modes = ['test']
    for mode in modes:  # 遍历训练集和测试集
        mode_path = os.path.join(dir_root, mode)
        for case in os.listdir(mode_path):  # 遍历所有case1-case10
            case_path = os.path.join(mode_path, case)
            img_save_path = case_path.replace('aff_4DCT', img_save_root)  # 设置image保存路径
            mask_save_path = case_path.replace('aff_4DCT', mask_save_root)  # 设置mask保存路径
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)

            mhd_file = glob(case_path + "/*.mhd")  # 读取文件
            for mhd in mhd_file:  # 遍历所有相位T00-T90
                print(mhd)
                _, file_name = os.path.split(mhd)
                # 读取.mhd文件
                mask_path = mhd.replace('aff_4DCT', "aff_mask")  # 查询mask路径
                sitk_img = sitk.ReadImage(mhd)
                sitk_mask = sitk.ReadImage(mask_path)
                # 读取源数据的信息
                spacing = sitk_img.GetSpacing()
                direction = sitk_img.GetDirection()

                img = sitk.GetArrayFromImage(sitk_img)
                mask = sitk.GetArrayFromImage(sitk_mask)
                # resize and normalize image & mask

                img, spacing = resize_numpy(img, value=-1024, spacing=list(spacing))
                img = normalize_numpy(img)
                mask, _ = resize_numpy(mask, value=0, spacing=list(spacing))
                # 转换numpy文件
                img = img.astype(np.float32)
                mask = mask.astype(np.short)
                new_img = sitk.GetImageFromArray(img)
                new_mask = sitk.GetImageFromArray(mask)
                # 设置转换后图像的信息
                new_img.SetSpacing(spacing)
                new_img.SetDirection(direction)
                new_mask.SetSpacing(spacing)
                new_mask.SetDirection(direction)

                # 保存转换后的.mhd文件
                sitk.WriteImage(new_img, os.path.join(img_save_path, file_name))
                sitk.WriteImage(new_mask, os.path.join(mask_save_path, file_name))


def generate_affine_image(img, mask, trans, translation=5, scale=(0.9, 1.1), rotation=1 / 36 * pi,
                          shear=1 / 36 * pi):
    img = torch.from_numpy(np.expand_dims(img, axis=[0, 1])).float()
    mask = torch.from_numpy(np.expand_dims(mask, axis=[0, 1])).float()
    # 随机产生变换参数

    translations = torch.rand(size=(1, 3), dtype=torch.float) * 2 * translation - translation
    scales = torch.rand(size=(1, 3), dtype=torch.float) * (scale[1] - scale[0]) + scale[0]
    rotations = torch.rand(size=(1, 3), dtype=torch.float) * 2 * rotation - rotation
    shears = torch.rand(size=(1, 6), dtype=torch.float) * 2 * shear - shear

    aff = (translations, rotations, scales, shears)
    # 使用transformer进行变换
    wraped_img = trans(aff, img, img)
    wraped_mask = trans(aff, mask, mask)
    wraped_img = np.asarray(torch.squeeze(wraped_img))
    wraped_mask = np.asarray(torch.squeeze(wraped_mask))
    return wraped_img, wraped_mask


def crop_center(img, cropz, cropx, cropy):
    z, x, y = img.shape
    startz = z // 2 - (cropz // 2)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[startz:startz + cropz, startx:startx + cropx, starty:starty + cropy]


def random_affine_image_train():
    train_root = '../4DCT/train'
    img_save_root = 'aff_4DCT'
    mask_save_root = 'aff_mask'
    # 随机产生两幅图像
    niter = 2
    aff_trans = AffineTransformer(ndim=3, coord_dim=1)
    case_list=os.listdir(train_root)
    #print(case_list)
    for case in case_list:  # 遍历所有case1-case10
        case_path = os.path.join(train_root, case)
        img_save_path = case_path.replace('4DCT', img_save_root)  # 设置转换后image保存路径
        mask_save_path = case_path.replace('4DCT', mask_save_root)  # 设置转换后mask保存路径
        # 创建文件夹
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        mhd_file = glob(case_path + "/*.mhd")  # 遍历所有mhd文件
        for mhd in mhd_file:  # 遍历所有相位T00-T90
            print(mhd)
            _, file_name = os.path.split(mhd)
            file_name = file_name[:-4]
            # 读取.mhd文件
            sitk_img = sitk.ReadImage(mhd)
            sitk_mask = sitk.ReadImage(mhd.replace('4DCT', 'mask'))
            # 读取源数据的信息
            spacing = sitk_img.GetSpacing()
            direction = sitk_img.GetDirection()

            img = sitk.GetArrayFromImage(sitk_img)
            mask = sitk.GetArrayFromImage(sitk_mask)

            shape = img.shape
            img = np.pad(img, pad_width=(100,), mode='constant', constant_values=(-2048,))
            mask = np.pad(mask, pad_width=(100,), mode='constant', constant_values=(0,))
            for i in range(niter):
                # 对原图以及肺掩码进行变换
                warped_img, warped_mask = generate_affine_image(img, mask, aff_trans)
                # 裁剪图像以及mask
                warped_img = crop_center(warped_img, *shape)
                warped_mask = crop_center(warped_mask, *shape)
                # 转换为short类型
                warped_mask = np.where(warped_mask > 0.4, 1, 0)
                warped_img = warped_img.astype(np.short)
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
                img_save_file = os.path.join(img_save_path, file_name + "_{}.mhd".format(i))
                mask_save_file = os.path.join(mask_save_path, file_name + "_{}.mhd".format(i))
                # print(img_save_file, mask_save_file)
                warped_img = sitk.GetImageFromArray(warped_img)
                warped_mask = sitk.GetImageFromArray(warped_mask)
                # 设置转换后图像的信息
                warped_img.SetSpacing(spacing)
                warped_img.SetDirection(direction)
                warped_mask.SetSpacing(spacing)
                warped_mask.SetDirection(direction)
                # 保存转换后的.mhd文件
                sitk.WriteImage(warped_img, img_save_file)
                sitk.WriteImage(warped_mask, mask_save_file)


def random_affine_image_test():
    test_root = '../4DCT/test'
    img_save_root = 'aff_4DCT'
    mask_save_root = 'aff_mask'
    landmark_path = '../parameter/landmark'
    aff_trans = AffineTransformer(ndim=3, coord_dim=1)
    #case_list = ['case8']
    case_list = os.listdir(test_root)
    for case in case_list:  # 遍历所有case1-case10
        case_path = os.path.join(test_root, case)
        img_save_path = case_path.replace('4DCT', img_save_root)  # 设置image保存路径
        mask_save_path = case_path.replace('4DCT', mask_save_root)  # 设置转换后mask保存路径
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)

        mhd_file = glob(case_path + "/*.mhd")  # 读取文件
        for mhd in mhd_file:  # 遍历所有相位T00、T50
            print(mhd)
            _, file_name = os.path.split(mhd)
            file_name = file_name[:-4]
            # 读取.mhd文件
            sitk_img = sitk.ReadImage(mhd)
            sitk_mask = sitk.ReadImage(mhd.replace('4DCT', 'mask'))
            # 读取源数据的信息
            spacing = sitk_img.GetSpacing()
            direction = sitk_img.GetDirection()

            img = sitk.GetArrayFromImage(sitk_img)
            mask = sitk.GetArrayFromImage(sitk_mask)
            shape = img.shape
            # 填充图像边缘100个体素
            # img = np.pad(img, pad_width=(100,), mode='constant', constant_values=(-1024,))
            # mask = np.pad(mask, pad_width=(100,), mode='constant', constant_values=(0,))
            # 生成随机转换
            if "T00" in mhd:
                landmark_file = os.path.join(landmark_path, "{}_300_T00_zxy.csv".format(case))
                dif = (128 - shape[0]) // 2
                df = pd.read_csv(landmark_file, header=None)
                # z = pd.DataFrame(df.iloc[:, 0]+dif)  # 变换z轴
                z = pd.DataFrame(df.iloc[:, 0])  # 变换z轴
                xy = pd.DataFrame(df.iloc[:, [1, 2]])
                # 拼接z,xy坐标
                df = pd.DataFrame(pd.concat([z, xy], axis=1))
                warped_img = img
                warped_mask = mask
                landmark_save_root = landmark_path.replace('landmark', 'new_landmark')
                if not os.path.exists(landmark_save_root):
                    os.makedirs(landmark_save_root)
                landmark_save_path = os.path.join(landmark_save_root, "{}_300_T00_zxy.csv".format(case))
                df.to_csv(landmark_save_path, index=False, header=None)
            else:
                landmark_file = os.path.join(landmark_path, "{}_300_T50_zxy.csv".format(case))
                print(landmark_file)
                # dif = (128 - shape[0]) // 2
                df = pd.read_csv(landmark_file, header=None)
                z = pd.DataFrame(df.iloc[:, 0])  # 变换z轴
                xy = pd.DataFrame(df.iloc[:, [1, 2]])
                # 拼接z,xy坐标
                df = pd.DataFrame(pd.concat([z, xy], axis=1))
                # 仅转换T50图像
                warped_img, warped_mask = generate_affine_image(img, mask, aff_trans)

                # E = torch.eye(3).unsqueeze(dim=0).float()
                # pad = torch.tensor(data=[[100], [100], [100]]).unsqueeze(dim=0).float()
                # translation = np.asarray(
                #     torch.squeeze(aff_trans.translation + torch.bmm(aff_trans.Tmat - E, pad).squeeze_(dim=-1), dim=0))
                translation = np.asarray(torch.squeeze(aff_trans.translation, dim=0))
                Tmat = np.asarray(torch.squeeze(aff_trans.Tmat, dim=0))
                print(Tmat)
                print(translation)
                Tmat = np.linalg.inv(Tmat)

                # TODO:convert landmark
                df = affine_landmark(Tmat, translation, df, shape)
                # z = pd.DataFrame(df.iloc[:, 0]+dif)  # 变换z轴
                z = pd.DataFrame(df.iloc[:, 0])  # 变换z轴
                xy = pd.DataFrame(df.iloc[:, [1, 2]])
                df = pd.DataFrame(pd.concat([z, xy], axis=1))
                landmark_save_root = landmark_path.replace('landmark', 'new_landmark')
                if not os.path.exists(landmark_save_root):
                    os.makedirs(landmark_save_root)
                landmark_save_path = os.path.join(landmark_save_root, "{}_300_T50_zxy.csv".format(case))
                df.to_csv(landmark_save_path, index=False, header=None)
            # 裁剪出图像原有的尺寸
            # warped_img = crop_center(warped_img, *shape)
            # warped_mask = crop_center(warped_mask, *shape)
            warped_mask = np.where(warped_mask > 0.4, 1, 0)
            warped_img = warped_img.astype(np.short)
            warped_mask = warped_mask.astype(np.short)
            # 使用闭操作进行处理
            warped_mask = sitk.GetImageFromArray(warped_mask)
            bm = sitk.BinaryMorphologicalClosingImageFilter()
            bm.SetKernelType(sitk.sitkBall)
            bm.SetKernelRadius(3)
            bm.SetForegroundValue(1)
            warped_mask = bm.Execute(warped_mask)
            warped_mask = sitk.GetArrayFromImage(warped_mask)
            # 创建img、mask存储文件名
            img_save_file = os.path.join(img_save_path, file_name + ".mhd")
            mask_save_file = os.path.join(mask_save_path, file_name + ".mhd")
            print(img_save_file, mask_save_file)
            warped_img = sitk.GetImageFromArray(warped_img)
            warped_mask = sitk.GetImageFromArray(warped_mask)
            # 设置转换后图像的信息
            warped_img.SetSpacing(spacing)
            warped_img.SetDirection(direction)
            warped_mask.SetSpacing(spacing)
            warped_mask.SetDirection(direction)
            # 保存转换后的.mhd文件
            sitk.WriteImage(warped_img, img_save_file)
            sitk.WriteImage(warped_mask, mask_save_file)


def affine_landmark(Tmat, translation, df, shape):
    new_df = pd.DataFrame()
    shape = np.asarray(shape) / 2
    # shape = shape[::-1]
    # print('shape', shape)
    for indexs in df.index:
        coor = np.asarray(list(df.loc[indexs].values))
        new_coor = np.matmul(Tmat, coor[::-1] - shape - translation) + shape
        new_coor = new_coor[::-1]
        new_df = new_df.append(
            pd.Series([round(new_coor[0], ndigits=4), round(new_coor[1], ndigits=4), round(new_coor[2], ndigits=4)]),
            ignore_index=True)
    return new_df


def convert_landmark(landmark_root, save_path='../parameter/landmark'):
    """
    landmark_root = r'H:\datasets\DIR-Lab\4DCT'
    原始Dir-lab数据形状为（x,y,z）,保存后的mhd文件为（z,x,y),需要对原始300个 landmark进行转换
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Phases = ['00', '50']
    for case in list(range(1, 11)):
        for P in Phases:
            csv_path = os.path.join(save_path, 'case' + str(case) + '_300_T' + P + "_zxy.csv")
            df = pd.DataFrame()
            print(csv_path)
            case_path = landmark_root + '/Case' + str(case) + 'Pack/'
            if case < 6:
                coor_path = case_path + "ExtremePhases/" + 'case' + str(case) + '_300_T' + P + "_xyz.txt"
            else:
                coor_path = case_path + "extremePhases/" + 'case' + str(case) + '_dirlab300_T' + P + "_xyz.txt"
            with open(coor_path) as f:
                for line in f.readlines():
                    line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                    str_coor = line.split("	")
                    coor = [int(str_coor[2]), int(str_coor[1]), int(str_coor[0])]
                    df = df.append(pd.Series(data=coor), ignore_index=True)
            df.to_csv(csv_path, index=False, header=False)


if __name__ == '__main__':
    img_path = '../4DCT_pro'
    mask_name = '../mask_pro'
    train_data = DIR_Lab(img_path, mask_name, mode='train')
    # print(len(train_data))
    print(train_data.image_pair_list)
    # random_affine_image_test()
    # resize_and_normalize_image(modes=['test'])
    # fixed, moving = train_data[0][0], train_data[0][1]
    # print(np.unique(moving))
    # train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    #
    # fixed_center = fixed[:, :, 64]
    # fixed_center = np.asarray(fixed_center).squeeze()
    #
    # moving_center = moving[:, :, 64]
    # moving_center = np.asarray(moving_center).squeeze()
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(fixed_center, cmap='gray')
    # plt.title('fixed')
    # plt.show()
    # plt.imshow(moving_center, cmap='gray')
    # plt.title('moving')
    # plt.show()
    # resize_and_normalize_image()
