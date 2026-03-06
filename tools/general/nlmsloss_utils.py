# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class MSloss(nn.Module):
    def __init__(self):
        super(MSloss, self).__init__()
        self.beta = 1e-7
        self.lambdaTV = 1e-8
        self.penalty = 'l1'

    def resize_for_tensors(self, tensors, size, mode='bilinear', align_corners=False):
        return F.interpolate(tensors, size, mode=mode, align_corners=align_corners)

    def levelsetLoss(self, output, target):
        outshape = output.shape  # [b,21,512,512]
        tarshape = target.shape  # [b, 3, 512 , 512]
        loss = 0.0
        for ich in range(tarshape[1]):  # 每个通道处

            target_ = torch.unsqueeze(target[:, ich], 1)  # [b,h,w] [b,1,h,w]
            target_ = target_.expand(
                tarshape[0],
                outshape[1],
                tarshape[2],
                tarshape[3])  # # [b,21,h,w]
            with torch.no_grad():
                pcentroid = torch.sum(
                    target_ * output, (2, 3)) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss) / outshape[0]

        return loss

    def gradientLoss2d(self, output):
        outshape = output.shape
        dH = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        dW = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        if (self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = (torch.sum(dH) + torch.sum(dW)) / outshape[0]
        return loss


    def nonlocal_total_variation_Loss(self, f_output, f_sem):

        """
        本函数计算网络前向输出的非局部梯度；
        像素r和像素Sn两点间的非局部导数为：两点差值dL乘以两点的约束系数tau；

        从output中取出像素r pixel_r_out，计算r和其它像素Sn之间的差dL_pixel_r，

        Args:
            f_output: DeepLabV3+:（16，21，128，128）；分割网络仅剩最后一步上采样操作的前向输出；但是根据msloss原文，这个前向输出需要经过softmax；
                    DeepLabV2:（16，21，64，64）；分割网络仅剩最后一步上采样操作的前向输出；但是根据msloss原文，这个前向输出需要经过softmax；
            f_sem: （16，256，32，32）；分割网络backbone的未经过上采样的输出，即语义特征f_i^s，为resnet101时输出

        Returns:

        """

        # 使用32*32尺寸计算NLTV；f_sem是32*32
        bs, _, H, W = f_sem.shape
        f_output = F.interpolate(f_output, (H, W), mode='bilinear', align_corners=True)   # f_output(16,21,32,32)
        f_sem_norm = F.normalize(f_sem, dim=1)  # f_sem_norm:(16,256,128,128)/(16,256,64,64)
        loss = 0

        # 从output中取出像素r pixel_r_out，计算r和其它像素Sn之间的差dL_pixel_r，
        for h in list(range(H)):
            for w in list(range(W)):
                pixel_r_out = f_output[:, :, h, w].unsqueeze(2).unsqueeze(3)        # pixel_r_output(16,21,1,1)
                pixel_r_sem = f_sem_norm[:, :, h, w].unsqueeze(2).unsqueeze(3)      # pixel_r_sem(16,256,1,1)

                # pixel r 和 其它像素Sn之间的差dL_pixel_r，即向量的模；可为一范数或二范数
                dL_pixel_r = torch.abs(pixel_r_out - f_output)      # dL_pixel_r:(16,21,128,128)/(16,21,64,64);
                if (self.penalty == "l2"):
                    dL_pixel_r = dL_pixel_r * dL_pixel_r

                # 计算像素r 和 其它像素Sn之间的约束系数tau
                tau_pixel_r = F.relu(torch.sum(pixel_r_sem * f_sem_norm, dim=1), inplace=True)   # tau_pixel_r:(16,128,128)/(16,64,64)；
                tau_pixel_r_max = tau_pixel_r.max(dim=1)[0].unsqueeze(1).max(dim=2)[0].unsqueeze(2)   # tau_pixel_r_max:(16,1,1)
                tau_pixel_r_norm = tau_pixel_r / (tau_pixel_r_max + 1e-5)            # tau_pixel_r_norm:(16,128,128)/(16,64,64)

                # 计算像素r处的非局部梯度
                nonlocal_gradient_pixel_r = torch.matmul(dL_pixel_r, tau_pixel_r_norm.unsqueeze(1))  #(16,21,128,128)/(16,21,64,64)

                loss_pixel_r = torch.sum(nonlocal_gradient_pixel_r) / bs    # 计算一个batch所有图像在像素r处的平均损失；
                loss += loss_pixel_r                                        # 计算一个batch所有图像所有像素产生的平均损失

        return loss


    def forward(self, output, target, f_sem, f_output, use_NonLocal_TV):
        """

        Args:
            output: 网络前向输出（batchsize，class_num, H,W）(16,21,512,512)
            target: 训练原始图像（batchsize，channel, H,W）（16，3，512，512）
            f_sem: backbone的输出，resnet101的输出为：DeepLabV3+（batchsize， 256，32，32），
                                                   DeepLabV2（batchsize， 256，64，64）
            f_output:   DeepLabV3+:（16，21，128，128）；分割网络仅剩最后一步上采样操作的前向输出；但是根据msloss原文，这个前向输出需要经过softmax；
                        DeepLabV2:（16，21，64，64）；分割网络仅剩最后一步上采样操作的前向输出；但是根据msloss原文，这个前向输出需要经过softmax；
            use_NonLocal_TV:是否使用带有非局部全变分项的MSloss

        Returns:

        """
        # output = self.resize_for_tensors(output, target.size()[2:], align_corners=True)
        loss_LS = self.levelsetLoss(output, target)  # 水平集LS（levelset）；和原图中的颜色有关

        if use_NonLocal_TV is True:
            loss_TV = self.nonlocal_total_variation_Loss(f_output, f_sem) * self.lambdaTV  # 非局部全变分项TV；梯度
            loss_MS = (loss_LS + loss_TV) * self.beta
        else:
            loss_TV = self.gradientLoss2d(output) * self.lambdaTV  # 全变分项TV；梯度
            loss_MS = (loss_LS + loss_TV) * self.beta

        return loss_LS * self.beta, loss_TV * self.beta, loss_MS





