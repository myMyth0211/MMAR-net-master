from math import pi
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import numpy as np
from models.utils import ScaledTanH, ScalingAF


class AffineNet(nn.Module):
    def __init__(self, in_dim=1, ratio=1, input_shape=(128, 128, 128),
                 drop=False, maxpool=True):
        super(AffineNet, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        self.input_shape = input_shape
        self.drop = drop
        self.maxpool = maxpool
        # 设置网络结构
        self.flow_dim = [2 * in_dim, 4, 8, 16, 32, 64]
        self.encoding = self.make_encoder()

        dense = [i // 2 ** 5 for i in input_shape]
        self.dense = np.prod(dense).item() * self.flow_dim[-1] * ratio  # 计算全连接层节点

        self.translation = nn.Linear(self.dense, 3)
        self.rotation = nn.Sequential(nn.Linear(self.dense, 3), ScaledTanH(self.max_rotation), )
        self.scaling = nn.Sequential(nn.Linear(self.dense, 3), ScalingAF(2))
        self.shearing = nn.Sequential(nn.Linear(self.dense, 2 * 3), ScaledTanH(self.max_shearing))

    def make_encoder(self):
        flow_dim = [2 * self.in_dim] + [dim * self.ratio for dim in self.flow_dim]

        encoder = []
        for i in range(len(flow_dim) - 1):
            encoder.append(Mutil_Branch_Module(flow_dim[i], flow_dim[i + 1]))
            if self.maxpool:
                encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                encoder.append(nn.Conv3d(flow_dim[i + 1], flow_dim[i + 1], kernel_size=2,
                                         stride=2, padding=0))
        encoder.append(nn.Conv3d(flow_dim[-1], flow_dim[-1], kernel_size=3,
                                 stride=1, padding=2))
        return nn.Sequential(*encoder)

    def forward(self, fixed, moving):

        x = torch.cat([fixed, moving], dim=1)
        feature = self.encoding(x)
        feature = torch.flatten(feature, 1)
        # 预测变换参数

        translation = self.translation(feature)
        rotation = self.rotation(feature)
        scale = self.scaling(feature)
        shear = self.shearing(feature)

        return translation, rotation, scale, shear


class Mutil_Branch_Module(nn.Module):
    def __init__(self, in_dim, out_dim, short=True):
        super(Mutil_Branch_Module, self).__init__()

        self.short = short
        self.conv = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True)
                                  )
        self.high_branch = nn.Sequential(nn.ConvTranspose3d(out_dim, out_dim // 2, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm3d(out_dim // 2), nn.ReLU(inplace=True),
                                         nn.Conv3d(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True),
                                         )
        self.norm_branch = nn.Sequential(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True),
                                         )
        self.low_branch = nn.Sequential(nn.Conv3d(out_dim, out_dim * 2, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm3d(out_dim * 2), nn.ReLU(inplace=True),
                                        nn.ConvTranspose3d(out_dim * 2, out_dim, kernel_size=3, stride=2, padding=1,
                                                           output_padding=1),
                                        nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True),
                                        )

    def forward(self, x):
        x = self.conv(x)
        high = self.high_branch(x)
        norm = self.norm_branch(x)
        low = self.low_branch(x)

        if self.short:
            x = high + norm + low + x
        else:
            x = high + norm + low
        x = F.relu(x)

        return x


class MutilStrideModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MutilStrideModule, self).__init__()

        self.scale1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale2 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale3 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=3, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(nn.Conv3d(3 * out_dim, out_dim, kernel_size=1),
                                  nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        size_x = scale2.shape[2:]
        scale1 = F.interpolate(scale1, size=size_x, mode='trilinear', align_corners=True)
        scale3 = F.interpolate(scale3, size=size_x, mode='trilinear', align_corners=True)

        x = torch.cat([scale1, scale2, scale3], dim=1)
        x = F.interpolate(self.conv(x), scale_factor=2, mode='trilinear', align_corners=True)
        return x


class MutilStrideModule_V2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MutilStrideModule_V2, self).__init__()

        self.scale1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale2 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale3 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=3, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(2 * out_dim, out_dim, kernel_size=1),
                                   nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(2 * out_dim, out_dim, kernel_size=1),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x):
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        size_2 = scale2.shape[2:]
        size_1 = scale1.shape[2:]
        scale3 = F.interpolate(scale3, size=size_2, mode='trilinear', align_corners=True)

        scale2 = F.interpolate(self.conv1(torch.cat([scale2, scale3], dim=1)), size=size_1, mode='trilinear',
                               align_corners=True)

        return self.conv2(torch.cat([scale1, scale2], dim=1))


class MutilStrideModule_V2_1(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MutilStrideModule_V2_1, self).__init__()

        self.scale1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale2 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale3 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=3, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(2 * out_dim,out_dim, kernel_size=1),
                                   nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(3 * out_dim, out_dim, kernel_size=1),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x):
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        size_2 = scale2.shape[2:]
        size_1 = scale1.shape[2:]
        scale3 = F.interpolate(scale3, size=size_2, mode='trilinear', align_corners=True)
        scale3_1 = F.interpolate(scale3, size=size_1, mode='trilinear', align_corners=True)

        scale2 = F.interpolate(self.conv1(torch.cat([scale2, scale3], dim=1)), size=size_1, mode='trilinear',
                               align_corners=True)

        return self.conv2(torch.cat([scale1, scale2, scale3_1], dim=1))


class MutilStrideModule_V3(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MutilStrideModule_V3, self).__init__()

        self.scale1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale2 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.scale3 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=3, padding=1),
                                    nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(out_dim, out_dim, kernel_size=1),
                                   nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(out_dim, out_dim, kernel_size=1),
                                   nn.BatchNorm3d(out_dim), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        size_2 = scale2.shape[2:]
        size_1 = scale1.shape[2:]
        scale3_2 = F.interpolate(scale3, size=size_2, mode='trilinear', align_corners=True)
        scale3_1 = F.interpolate(scale3, size=size_1, mode='trilinear', align_corners=True)
        scale2 = F.interpolate(self.conv1(scale2 + scale3_2), size=size_1, mode='trilinear',
                               align_corners=True)

        return self.conv2(scale1 + scale2 + scale3_1)


class RegistrationNet(nn.Module):
    def __init__(self, in_dim=2, depth=5, ms=True, out_dim=None, scale=True):
        super(RegistrationNet, self).__init__()
        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        self.depth = depth
        self.scale = scale
        self.flow_dim = [in_dim, 8, 16, 32, 64, 128, 256]
        # self.flow_dim = [in_dim, 8, 16, 32, 32, 64, 64]
        # self.flow_dim = [in_dim, 8, 16, 32, 64, 64, 64]
        # self.flow_dim = [in_dim,16, 16, 32, 32, 64, 64, 128,128]
        if out_dim is None:
            self.out_dim = self.flow_dim[self.depth]
        else:
            self.out_dim = out_dim
        # assert len(self.flow_dim) == self.depth + 1
        # 核心模块
        self.encoder_list = nn.ModuleList()
        # self.down_list = nn.ModuleList()
        for i in range(depth):
            if ms:
                self.encoder_list.append(MutilStrideModule_V2_1(self.flow_dim[i], self.flow_dim[i + 1]))
            else:
                self.encoder_list.append(Conv_Bn_Relu(self.flow_dim[i], self.flow_dim[i + 1]))
            # if i < self.depth - 1:
            #     self.down_list.append(nn.Conv3d(self.flow_dim[i + 1], self.flow_dim[i + 1], kernel_size=2, stride=2))
        self.FAM = nn.ModuleList()
        self.start = 0
        assert self.start < depth
        if self.scale:
            for i in range(self.start, depth):
                self.FAM.append(nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                              nn.Conv3d(self.flow_dim[i + 1], self.out_dim, kernel_size=1),
                                              nn.LeakyReLU(inplace=True)
                                              ))
        else:
            self.FAM.append(nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                          nn.Conv3d(self.flow_dim[depth], self.out_dim, kernel_size=1),
                                          nn.LeakyReLU(inplace=True)
                                          ))
        # 输出层
        dense = self.out_dim
        self.regression = []

        for rl in range(2):
            self.regression.extend([nn.Linear(dense, dense), nn.LeakyReLU(inplace=True)])
        self.regression = nn.Sequential(*self.regression)

        self.translation = nn.Linear(dense, 3)
        self.rotation = nn.Sequential(nn.Linear(dense, 3), ScaledTanH(self.max_rotation), )
        self.scaling = nn.Sequential(nn.Linear(dense, 3), ScalingAF(2))
        self.shearing = nn.Sequential(nn.Linear(dense, 2 * 3), ScaledTanH(self.max_shearing))

    def forward(self, fixed, moving):
        scales_out = []
        x = torch.cat([fixed, moving], dim=1)
        for i, encoder in enumerate(self.encoder_list):
            x = encoder(x)
            if self.scale:
                if i >= self.start:
                    scales_out.append(torch.flatten(self.FAM[i - self.start](x), start_dim=1))
            else:
                if i + 1 == self.depth:
                    scales_out.append(torch.flatten(self.FAM[0](x), start_dim=1))
            # if i < self.depth - 1:
            #     x = self.down_list[i](x)
            if i < 2:
                x = F.max_pool3d(x, 2, 2)
            else:
                x = F.avg_pool3d(x, 2, 2)

        if self.scale:
            feature = attention_block_v5(scales_out)
        else:
            feature = scales_out[-1]

        # MLP进行回归
        feature = self.regression(feature)
        # 仿射参数预测
        translation = self.translation(feature)
        rotation = self.rotation(feature)
        scale = self.scaling(feature)
        shear = self.shearing(feature)

        return translation, rotation, scale, shear


def average_block(tensor_list):
    return torch.sum(torch.stack(tensor_list, dim=-1), dim=-1) / len(tensor_list)


def attention_block(tensor_list):
    x = torch.stack(tensor_list, dim=-1)
    x_ave = torch.sum(x, dim=-1, keepdim=True) / len(tensor_list)
    sig_ave = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_ave), dim=1)

    return torch.squeeze(torch.bmm(x, sig_ave), dim=-1)


def attention_block_v2(tensor_list):
    x = torch.stack(tensor_list, dim=-1)
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x_min, _ = torch.min(x, dim=-1, keepdim=True)

    sig_min = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_min), dim=1)
    sig_max = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_max), dim=1)

    return torch.squeeze(torch.bmm(x, sig_min) + torch.bmm(x, sig_max), dim=-1)


def attention_block_v3(tensor_list):
    x = torch.stack(tensor_list, dim=-1)
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x_min, _ = torch.min(x, dim=-1, keepdim=True)
    x_ave = torch.sum(x, dim=-1, keepdim=True) / len(tensor_list)

    sig_min = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_min), dim=1)
    sig_max = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_max), dim=1)
    sig_ave = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_ave), dim=1)

    return torch.squeeze(0.2 * torch.bmm(x, sig_min) + 0.2 * torch.bmm(x, sig_max) + 0.6 * torch.bmm(x, sig_ave),
                         dim=-1)
    # return torch.squeeze(torch.bmm(x, sig_min) +  torch.bmm(x, sig_max) + torch.bmm(x, sig_ave),dim=-1)


def attention_block_v4(tensor_list):
    n = len(tensor_list)
    x = torch.stack(tensor_list, dim=-1)
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    x_min, _ = torch.min(x, dim=-1, keepdim=True)
    # x_ave = torch.sum(x, dim=-1, keepdim=True) / n

    sig_min = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_min), dim=1)
    sig_max = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_max), dim=1)
    # sig_ave = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_ave), dim=1)

    return torch.squeeze(torch.bmm(x, 1 - sig_min) / (n - 1) + torch.bmm(x, sig_max), dim=-1)


def attention_block_v5(tensor_list):
    x = torch.stack(tensor_list, dim=-1)
    x_max, _ = torch.max(x, dim=-1, keepdim=True)
    # x_ave = torch.sum(x, dim=-1, keepdim=True) / len(tensor_list)

    sig_max = torch.softmax(torch.bmm(torch.transpose(x, 1, 2), x_max), dim=1)
    # print(sig_max)
    return torch.squeeze(torch.bmm(x, sig_max), dim=-1)


class FlashReg(nn.Module):

    def __init__(self, in_dim=2, depth=5):
        super(FlashReg, self).__init__()
        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi
        self.depth = depth
        self.flow_dim = [in_dim, 8, 16, 32, 64, 128, 256]

        self.encoder = self.make_encoder()
        dense = self.flow_dim[depth]
        self.regression = []
        for rl in range(2):
            self.regression.extend([nn.Linear(dense, dense), nn.LeakyReLU(inplace=True)])
        self.regression = nn.Sequential(*self.regression)

        self.translation = nn.Linear(dense, 3)
        self.rotation = nn.Sequential(nn.Linear(dense, 3), ScaledTanH(self.max_rotation), )
        self.scaling = nn.Sequential(nn.Linear(dense, 3), ScalingAF(2))
        self.shearing = nn.Sequential(nn.Linear(dense, 2 * 3), ScaledTanH(self.max_shearing))

    def make_encoder(self):
        encoder = []
        for i in range(self.depth):
            encoder.extend(
                [nn.Conv3d(self.flow_dim[i], self.flow_dim[i + 1], kernel_size=3, padding=1, padding_mode='replicate'),
                 nn.BatchNorm3d(self.flow_dim[i + 1]), nn.LeakyReLU(inplace=True)])
            encoder.append(FLASHTransformer(in_channel=self.flow_dim[i + 1], path_size=1, emb_size=self.flow_dim[i + 1],
                                            dim=self.flow_dim[i + 1], depth=12, group_size=self.flow_dim[i + 1],
                                            query_key_dim=self.flow_dim[i + 1] // 2, img_size=128 // (i + 1)))
            if i != self.depth - 1:
                encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                encoder.append(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        return nn.Sequential(*encoder)

    def forward(self, fixed, moving):
        x = torch.cat([fixed, moving], dim=1)
        feature = self.encoder(x)
        # MLP进行回归
        feature = self.regression(feature)
        # 仿射参数预测
        translation = self.translation(feature)
        rotation = self.rotation(feature)
        scale = self.scaling(feature)
        shear = self.shearing(feature)

        return translation, rotation, scale, shear


class RegistrationNet_v2(nn.Module):

    def __init__(self, in_dim=1, depth=5, maxpool=False, ms=True, scale=True, out_dim=None):
        super(RegistrationNet_v2, self).__init__()
        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        self.maxpool = maxpool
        self.depth = depth
        self.scale = scale

        self.flow_dim = [in_dim, 8, 16, 32, 64, 128, 256, 512]

        if out_dim is None:
            self.out_dim = self.flow_dim[self.depth]
        else:
            self.out_dim = out_dim
        # 核心模块
        self.encoder_list = nn.ModuleList()
        for i in range(depth):
            if ms:
                self.encoder_list.append(MutilStrideModule_V2(self.flow_dim[i], self.flow_dim[i + 1]))
            else:
                self.encoder_list.append(Conv_Bn_Relu(self.flow_dim[i], self.flow_dim[i + 1]))
        self.FAM = nn.ModuleList()
        if self.scale:
            for i in range(depth):
                self.FAM.append(nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                              nn.Conv3d(self.flow_dim[i + 1], self.out_dim, kernel_size=1),
                                              nn.LeakyReLU(inplace=True)
                                              ))
        else:
            self.FAM.append(nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                          nn.Conv3d(self.flow_dim[depth], self.out_dim, kernel_size=1),
                                          nn.LeakyReLU(inplace=True)
                                          ))
        layers_depth = 2
        dense = self.flow_dim[depth] * 2
        # dense = self.flow_dim[depth]
        self.regression = []
        for l in range(layers_depth):
            self.regression.extend([nn.Linear(dense, dense), nn.LeakyReLU(inplace=True)])
        self.regression = nn.Sequential(*self.regression)

        self.translation = nn.Linear(dense, 3)
        self.rotation = nn.Sequential(nn.Linear(dense, 3), ScaledTanH(self.max_rotation), )
        self.scaling = nn.Sequential(nn.Linear(dense, 3), ScalingAF(2))
        self.shearing = nn.Sequential(nn.Linear(dense, 2 * 3), ScaledTanH(self.max_shearing))

    def forward(self, fixed, moving):
        # 提取图像特征
        scales_out = []
        fixed_x = fixed
        moving_x = moving
        for i, encoder in enumerate(self.encoder_list):
            fixed_x = encoder(fixed_x)
            moving_x = encoder(moving_x)
            # print(fixed_x.shape)
            if self.scale:
                scales_out.append(torch.cat([torch.flatten(self.FAM[i](fixed_x), start_dim=1),
                                             torch.flatten(self.FAM[i](moving_x), start_dim=1)], dim=1))
            else:
                if i + 1 == self.depth:
                    scales_out.append(torch.cat([torch.flatten(self.FAM[0](fixed_x), start_dim=1),
                                                 torch.flatten(self.FAM[0](moving_x), start_dim=1)], dim=1))

            if i < 2:
                fixed_x = F.max_pool3d(fixed_x, 2, 2)
                moving_x = F.max_pool3d(moving_x, 2, 2)
            else:
                fixed_x = F.avg_pool3d(fixed_x, 2, 2)
                moving_x = F.avg_pool3d(moving_x, 2, 2)

        if self.scale:
            feature = attention_block_v2(scales_out)
        else:
            feature = scales_out[-1]
        # 参数回归
        feature = self.regression(feature)
        # 参数预测
        translation = self.translation(feature)
        rotation = self.rotation(feature)
        scale = self.scaling(feature)
        shear = self.shearing(feature)

        return translation, rotation, scale, shear


class RegistrationNet_v3(nn.Module):

    def __init__(self, indim=1, depth=(3, 3), input_shape=(128, 128, 128)):
        super(RegistrationNet_v3, self).__init__()
        self.max_scaling = 2
        self.max_rotation = 0.5 * pi
        self.max_shearing = 0.25 * pi

        self.in_dim = indim
        self.depth = depth
        self.flow_dim_1 = [indim, 4, 8, 16, 32, 64, 128]
        self.feature_dim = [self.flow_dim_1[self.depth[0]] * 2, 64, 64, 64, 64]
        # 构建特征提取层
        self.encoder = self.make_encoder_module()
        # 构建参数回归层
        self.regression = self.mask_regression_module()
        # 全连接层预测仿射参数
        dense = [i // 2 ** sum(depth) for i in input_shape]
        # self.dense = np.prod(dense).item() * self.feature_dim*2**self.depth[1]
        self.dense = np.prod(dense).item() * self.feature_dim[self.depth[1]]
        self.translation = nn.Linear(self.dense, 3)
        self.rotation = nn.Sequential(nn.Linear(self.dense, 3), ScaledTanH(self.max_rotation), )
        self.scaling = nn.Sequential(nn.Linear(self.dense, 3), ScalingAF(2))
        self.shearing = nn.Sequential(nn.Linear(self.dense, 2 * 3), ScaledTanH(self.max_shearing))

    def forward(self, fixed, moving):
        fixed_feature = self.encoder(fixed)
        moving_feature = self.encoder(moving)
        feature = torch.cat([fixed_feature, moving_feature], dim=1)
        feature = self.regression(feature)

        feature = torch.flatten(feature, 1)
        # 预测参数

        translation = self.translation(feature)
        rotation = self.rotation(feature)
        scale = self.scaling(feature)
        shear = self.shearing(feature)

        return translation, rotation, scale, shear

    def make_encoder_module(self):
        module = []
        for i in range(self.depth[0]):
            module.append(Conv_Bn_Relu(self.flow_dim_1[i], self.flow_dim_1[i + 1]))
            module.append(nn.MaxPool3d(kernel_size=2, stride=2))
        # module.extend([nn.Conv3d(self.flow_dim_1[self.depth[0]], self.flow_dim_1[self.depth[0]],
        #                         kernel_size=3, stride=1, padding=1),
        #               nn.BatchNorm3d(self.flow_dim_1[self.depth[0]]), nn.LeakyReLU(inplace=True)]
        #              )
        return nn.Sequential(*module)

    def mask_regression_module(self):
        block = []
        for j in range(self.depth[1]):
            block.append(nn.Conv3d(self.feature_dim[j], self.feature_dim[j + 1],
                                   kernel_size=3, stride=1, padding=1))
            block.append(nn.BatchNorm3d(self.feature_dim[j + 1]))
            block.append(nn.LeakyReLU(inplace=True))
            block.append(nn.Conv3d(self.feature_dim[j + 1], self.feature_dim[j + 1],
                                   kernel_size=2, stride=2, padding=0))
        # block.extend([nn.Conv3d(self.feature_dim[self.depth[1]], self.feature_dim[self.depth[1]],
        #                         kernel_size=3, stride=1, padding=1),
        #               nn.BatchNorm3d(self.feature_dim[self.depth[1]]), nn.LeakyReLU(inplace=True)]
        #              )
        return nn.Sequential(*block)


class Conv_Bn_Relu(nn.Module):

    def __init__(self, indim, outdim):
        super(Conv_Bn_Relu, self).__init__()
        block = []
        block.append(nn.Conv3d(indim, outdim, kernel_size=3, stride=1, padding=1))
        block.append(nn.BatchNorm3d(outdim))
        block.append(nn.LeakyReLU(inplace=True))
        # block.append(nn.Conv3d(outdim, outdim, kernel_size=3, stride=1, padding=1))
        # block.append(nn.BatchNorm3d(outdim))
        # block.append(nn.LeakyReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    # fixed = torch.randn(size=(2, 1, 128, 128, 128)).float()
    # moving = torch.randn(size=(2, 1, 128, 128, 128)).float()
    model = RegistrationNet(depth=6, ms=False, scale=False)
    print(model)
    total = sum([para.nelement() for para in model.parameters()])
    print("Numberof parameter: % .2fK" % (total / 1024))
    # y = model(fixed, moving)
    # for y0 in y:
    #     print(y0.shape)
