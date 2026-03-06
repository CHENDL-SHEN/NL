# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class PNRloss(nn.Module):
    def __init__(self, num_classes=21, feature_upsample=False):
        super(PNRloss, self).__init__()
        self.num_cls = num_classes
        self.feature_upsample = feature_upsample

    def one_hot_form(self, label):
        """
        本def是将 语义分割标签label变成one-hot形式
        标签label：(b, 1, h, w)或(b, h, w)
        输出one-hot形式的label：(b, classes, h, w)
        """
        num_classes = self.num_cls  # 分类类别数
        if len(label.shape) == 4:
            current_label = label.squeeze(1)  # （batch_size, 1, h, w) ---> （batch_size, h, w)
        else:
            current_label = label
        batch_size, h, w = current_label.shape[0], current_label.shape[1], current_label.shape[2]
        # print(h, w, batch_size)

        one_hots = []
        for i in range(num_classes):
            tmplate = torch.zeros(batch_size, h, w)  # （batch_size, h, w)
            tmplate[current_label == i] = 1
            tmplate = tmplate.view(batch_size, 1, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)

            one_hots.append(tmplate)
        onehot = torch.cat(one_hots, dim=1)

        return onehot

    def extract_image_prototype(self, one_hot_label, f_sem, f_hie):
        """
        3.B.1) 图像原型特征p_i提取；

        伪标签label是正常尺寸（在这里是经过data augment的，大小是512*512），f_sem和f_hie是下采样后的尺寸 32*32；
        所以伪标签label做索引指明 f_sem和f_hie 的前景背景区域时，这两者的尺寸是对不上的；
        因此本文将伪标签下采样到和特征图一样大，然后再生成原型;，这也是ProDA方法中的做法；

        fh=32，fw=32，h=512，w=512

        Args:
            label: (batchsize,21,512,512)
            f_sem: (batchsize,256,32,32)；ASPP层的输出；借鉴 BANA-stage3_vgg.py 和 ProDA方法；
            f_hie: (batchsize,256,32,32)
            imgH:  512；原图高
            imgW:  512；原图宽

        Returns:
            prototypes: (batchsize,21,256,1,1)

        """

        # 伪标签下采样到和 特征图f_sem、f_hie 一样尺寸，生成原型；
        n_sem, c_sem, h_sem, w_sem = f_sem.shape     # n_sem=1, c_sem=256, h_sem=281, w_sem=500
        n_hie, c_hie, h_hie, w_hie = f_hie.shape     # n_hie=1, c_hie=256, h_hie=281, w_hie=500

        label_fg = one_hot_label[:, 1:].cuda()  # label_fg:(1,1,281,500)
        label_bg = one_hot_label[:, 0].unsqueeze(1).cuda()  # label_fg:(1,20,281,500)

        index_fg = F.interpolate(label_fg, f_sem.shape[2:], mode='nearest')
        index_bg = F.interpolate(label_bg, f_hie.shape[2:], mode='nearest')

        tmp_sem = index_fg.unsqueeze(2) * f_sem.unsqueeze(1) # index_fg:[n,20,1,fh,fw], f_sem:[n,1,c,fh,fw], tmp_sem:[n,20,c,fh,fw]
        tmp_hie = index_bg.unsqueeze(2) * f_hie.unsqueeze(1) # index_bg:[n,1,1,fh,fw], f_hie:[n,1,c,fh,fw], tmp_hie:[n,1,c,fh,fw]

        prototype_sem = F.adaptive_avg_pool2d(tmp_sem.view(-1, c_sem, h_sem, w_sem), (1, 1)).view(n_sem, self.num_cls-1, c_sem, 1, 1)  # prototypes_sem:[n,20,c,1,1]
        prototype_hie = F.adaptive_avg_pool2d(tmp_hie.view(-1, c_hie, h_hie, w_hie), (1, 1)).view(n_hie, 1, c_hie, 1, 1)  # prototypes_hie:[n,1,c,1,1]

        prototypes = torch.cat([prototype_sem, prototype_hie], dim=1)       # prototypes_hie:[n,21,c,1,1]

        return prototypes

    def generate_pixel_level_weight(self, f_sem, prototypes, pselabel_one_hot, imgH, imgW, sigma):
        """
        3.B.2）伪标注像素级权重omiga的生成

        Args:
            f_sem: 训练图像由backbone表征的embedding特征；也即语义特征；未经过上采样；（b，c，fH，fW）；（16，256，32，32）（backbone为resnet101）
            prototypes: 所有类别（含背景）的原型特征；（b，class_num, c，1,1）;(16，21，256，1，1)
            pselabel_one_hot: one_hot形式的伪标注,即由（16，512，512）值为0-20的数，变成（16，21，512，512）值为0或1；
            imgH: 原图的高；puzzlecam代码中，原图和伪标注统一经过augment变成了512*512，所以imgH为512
            imgW: 原图的宽；512
            sigma: 权重omiga公式中的缩放参数，趋于无穷大时，sigma值被二值化为0或1，此时像素级加权的策略变为一种硬约束筛选策略；

        Returns:
            omiga：(b,H,W)；（16，512，512）；每个像素的伪标注的可信任程度；

        """

        bs, _, fh, fw, = f_sem.shape
        similarity = torch.zeros((bs, self.num_cls, fh, fw))
        normed_f_sem = F.normalize(f_sem, dim=1)   # (b,dims,fH,fW)=(16,256,32,32)
        normed_pro = F.normalize(prototypes, dim=1)  # (b, num_cls, dims, 1, 1)=(16,21,256,1,1)
        for i in range(0, bs):   # batchsize 中的每个图像
            similarity_i = F.conv2d(normed_f_sem[i].unsqueeze(0), normed_pro[i])  # (1, dims, fH, fW)*(num_cls, dims, 1, 1)=(num_cls,fH,fW);(1,256,32,32)*(21,256,1,1)=(21,32,32)
            similarity[i] = similarity_i

        similarity_us = F.interpolate(similarity, (imgH, imgW), mode='bilinear', align_corners=True)    # (b,c,H,W)=(16,21,512,512)；上采样到原图大小的similarity
        distance = similarity_us + 1
        max_distance = distance.max(dim=1)[0].unsqueeze(1)  # (b,num_cls,H,W)->(b,H,W)->(b,1,H,W);(16,1,512,512)
        omiga_ac = (distance / (max_distance + 1e-7)) ** sigma  # (b,num_cls,H,W)/(b,1,H,W)=(b,num_cls,H,W);(16,21,512,512)
        omiga = (omiga_ac * pselabel_one_hot).sum(dim=1) # (b,num_cls,H,W)*(b,num_cls,H,W)=(b,num_cls,H,W)-->(b,H,W);(16,512,512)

        return omiga

    def compute_adaptive_weighted_loss(self, output, label, omiga):

        """
        3.B.3）自适应加权损失计算
        Args:
            output: （16，21，512，512）；分割网络经过softmax的前向输出；输出需要经过上采样
            label:（16，512，512）；伪标注
            omiga: （16，512，512）；伪标注的像素级权重
            use_PNR_plus_Loss: 是否使用PNR_Loss的plus版本；区别是PNR_Loss的plus版本仅计算低置信伪标注带来的损失，
                               PNR_Loss利用omiga做归一化后得到plus版本

        Returns:

        """
        # 3.B.3）自适应加权损失计算
        # nn.CrossEntropyLoss函数输入标签索引时（即0-20的数字，而不是one-hot形式），标签必须是LongTensor；
        b, h, w = label.shape
        class_loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
        ce_loss = class_loss_fn(output, label.long())
        wce_loss = ce_loss * omiga.cuda()
        pnr_loss = torch.sum(wce_loss) / (b * h * w * torch.sum(omiga) + 1e-7 )
        pnr_loss[torch.isnan(pnr_loss)] = 0

        return pnr_loss

    def compute_adaptive_weighted_SC_loss(self, output, label, omiga, SC):

        """
        3.B.3）自适应加权损失计算
        Args:
            output: （16，21，512，512）；分割网络经过softmax的前向输出；输出需要经过上采样
            label:（16，512，512）；伪标注
            omiga: （16，512，512）；伪标注的像素级权重
            use_PNR_plus_Loss: 是否使用PNR_Loss的plus版本；区别是PNR_Loss的plus版本仅计算低置信伪标注带来的损失，
                               PNR_Loss利用omiga做归一化后得到plus版本

        Returns:

        """
        # 3.B.3）自适应加权损失计算
        # nn.CrossEntropyLoss函数输入标签索引时（即0-20的数字，而不是one-hot形式），标签必须是LongTensor；
        b, h, w = label.shape
        class_loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
        loss_t = 0

        for i in range(b):
            output_c = output[i, :, :, :].unsqueeze(0)
            label_c = label[i, :, :].unsqueeze(0)
            omiga_c = omiga[i, :, :].unsqueeze(0)
            loss_c = class_loss_fn(output_c, label_c.long()) # logits:Tensor(16,21,512,512) labels:Tensor(16,512,512)
            loss_c = (torch.sum(loss_c * omiga_c.cuda()) * float(SC[i])) / (h * w * torch.sum(omiga) + 1e-7)
            loss_c[torch.isnan(loss_c)] = 0
            loss_t += loss_c
            print("loss_c={} SC={}".format(loss_c, SC[i]))

        pnr_loss = loss_t / (b + 1e-5)

        return pnr_loss


    def forward(self, output, f_sem, f_hie, imgH, imgW, label, sigma, SC=None):
        """

        Args:
            output: （16，21，512，512）;分割网络经过softmax的前向输出
            f_sem:（16，256，32，32）;分割网络backbone提取的原图语义特征
            f_hie:（16，256，32，32);提取的原图浅层特征
            imgH: 原图高度
            imgW: 原图宽度
            label: (16，512，512)伪标注
            sigma: 伪标注权重omiga公式中的缩放参数
            use_PNR_plus_Loss: True or False

        Returns:

        """
        pselabel_one_hot = self.one_hot_form(label)
        prototypes = self.extract_image_prototype(pselabel_one_hot, f_sem, f_hie)
        omiga = self.generate_pixel_level_weight(f_sem, prototypes, pselabel_one_hot, imgH, imgW, sigma)
        if SC is not None:
            loss = self.compute_adaptive_weighted_SC_loss(output, label, omiga, SC)
        else:
            loss = self.compute_adaptive_weighted_loss(output, label, omiga)

        return loss


# if __name__ == '__main__':
#
#     import time
#     import datetime
#
#     out = torch.randn(16, 21, 512, 512).cuda()  # 产生形状为(16,21, 281, 500)的随机数tensor；正态分布
#     print("out", out)
#     label = torch.randint(0, 20, (16, 512, 512)).cuda()   # numpy中产生0-20的形状为(16,281,500)的ndarray数组
#     print("label", label)
#     f_sem = torch.randn(16, 256, 32, 32).cuda()
#     print("f_sem", f_sem)
#     f_hie = torch.randn(16, 256, 32, 32).cuda()
#     print("f_hie", f_hie)
#
#     imgH = 512
#     imgW = 512
#     use_PNR_plus_Loss = True
#
#     if use_PNR_plus_Loss:   # PNR_plus_loss，pnr loss作为ce loss的正则项
#         sigma = 7
#     else:
#         sigma = 1
#
#     loss = PNRloss(21, False)
#     pnr_loss = loss(out, f_sem, f_hie, imgH, imgW, label, sigma)
#     print(pnr_loss)
#
#     start = time.time()
#
#     end = time.time()
#
#     print(end-start)
