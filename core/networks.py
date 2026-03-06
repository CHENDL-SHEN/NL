# Copyright (C) 2021 * Ltd. All rights reserved.
# author: Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model
from .deeplab_utils import ASPP, Decoder

from tools.ai.torch_utils import resize_for_tensors


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=self.eps
        )


def group_norm(features):
    return nn.GroupNorm(4, features)


class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.num_cls = num_classes
        self.mode = mode

        if self.mode == 'fix':
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d

        if 'resnet' in model_name:
            self.model = resnet.ResNet(
                resnet.Bottleneck,
                resnet.layers_dic[model_name],
                strides=(2, 2, 2, 1),
                batch_norm_fn=self.norm_fn
            )

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.model.load_state_dict(state_dict)

        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(
                pretrained=True,
                dilated=dilated,
                dilation=dilation,
                norm_layer=self.norm_fn
            )

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)


class DeepLabv3_Plus(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn

        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)

    def forward(self, x):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x

        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.aspp(x)
        x = self.decoder(x, x_low_level)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x


class DeepLabv3_Plus_PMS(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn

        self.side2 = nn.Conv2d(256, 32, 1, bias=False)
        self.side3 = nn.Conv2d(512, 32, 1, bias=False)
        self.side4 = nn.Conv2d(1024, 96, 1, bias=False)
        self.side5 = nn.Conv2d(2048, 96, 1, bias=False)

        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)

    def get_hie_feature(self, x2, x3, x4, x5):
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())
        side5 = self.side5(x5.detach())

        f_hie = torch.cat([
            F.interpolate(
                side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5),
                side4.shape[2:],
                mode='bilinear'
            ),
            F.interpolate(
                side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5),
                side4.shape[2:],
                mode='bilinear'
            ),
            F.interpolate(
                side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5),
                side4.shape[2:],
                mode='bilinear'
            ),
            F.interpolate(
                side5 / (torch.norm(side5, dim=1, keepdim=True) + 1e-5),
                side4.shape[2:],
                mode='bilinear'
            )
        ], dim=1)

        return f_hie

    def forward(self, x):
        inputs = x

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x_low_level = x2

        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        x6 = self.aspp(x5)
        x7 = self.decoder(x6, x_low_level)

        output = resize_for_tensors(x7, inputs.size()[2:], align_corners=True)
        f_sem = x6
        f_hie = self.get_hie_feature(x2, x3, x4, x5)
        f_output = x7

        return output, f_sem, f_hie, f_output