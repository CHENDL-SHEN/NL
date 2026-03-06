# Copyright (C) 2020 * Ltd. All rights reserved.
# author: Sanghyeon Jo <josanghyeokn@gmail.com>

"""
Inference script for semantic segmentation.

Supported models:
    1) DeepLabv3_Plus_PMS
    2) DeepLabv3_Plus

This version supports:
    - multi-scale inference
    - horizontal flip inference
    - optional dense CRF post-processing
"""

import os
import sys
import copy
import argparse
import numpy as np
import imageio
import PIL

import torch
import torch.nn.functional as F

from core.puzzle_utils import crf_inference
from core.networks import DeepLabv3_Plus_PMS, DeepLabv3_Plus
from core.datasets import VOC_Dataset_For_Evaluation

from tools.general.io_utils import create_directory
from tools.general.time_utils import Timer
from tools.general.json_utils import read_json

from tools.ai.torch_utils import (
    load_model,
    calculate_parameters,
    get_numpy_from_tensor,
    str2bool
)

from tools.ai.evaluate_utils import resize_for_tensors
from tools.ai.augment_utils import Normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_dir', default='/media/ders/XS/dataset/VOC2012/', type=str)
    parser.add_argument('--domain', default='test', type=str)

    # Network
    parser.add_argument('--architecture', default='DeepLabv3+', type=str)
    parser.add_argument('--backbone', default='resnest101', type=str)
    parser.add_argument('--mode', default='fix', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)
    parser.add_argument('--model_type', default='PMS', type=str, choices=['PMS', 'V3+'])

    # Inference
    parser.add_argument('--tag', default='DeepLabv3+@ResNeSt-101@Fix@GN', type=str)
    parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
    parser.add_argument('--iteration', default=0, type=int)

    return parser


def build_model(args, num_classes):
    if args.model_type == 'PMS':
        model = DeepLabv3_Plus_PMS(
            args.backbone,
            num_classes=num_classes,
            mode=args.mode,
            use_group_norm=args.use_gn
        )
    elif args.model_type == 'V3+':
        model = DeepLabv3_Plus(
            args.backbone,
            num_classes=num_classes,
            mode=args.mode,
            use_group_norm=args.use_gn
        )
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    return model


def inference(model, images, image_size, model_type):
    images = images.cuda(non_blocking=True)

    if model_type == 'PMS':
        logits, _, _, _ = model(images)
    elif model_type == 'V3+':
        logits = model(images)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    logits = resize_for_tensors(logits, image_size)

    # Merge original and flipped predictions
    logits = logits[0] + logits[1].flip(-1)
    logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))

    return logits


def build_prediction_dir(base_tag, domain, scales, iteration):
    experiment_tag = f'{base_tag}@{domain}@scale={scales}@iteration={iteration}'

    if iteration > 0:
        pred_dir = create_directory(f'./experiments/predictions/dCRF@{experiment_tag}/')
    else:
        pred_dir = create_directory(f'./experiments/predictions/{experiment_tag}/')

    return pred_dir, experiment_tag


def main():
    args = get_argparser().parse_args()

    # Paths
    model_dir = create_directory('./experiments/models/')
    model_path = os.path.join(model_dir, f'{args.tag}.pth')

    pred_dir, experiment_tag = build_prediction_dir(
        args.tag, args.domain, args.scales, args.iteration
    )

    print(f'[i] model path: {model_path}')
    print(f'[i] prediction dir: {pred_dir}')
    print(f'[i] experiment tag: {experiment_tag}')
    print()

    # Normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    # Dataset metadata
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)

    # Build model
    model = build_model(args, meta_dic['classes'] + 1)
    model = model.cuda()
    model.eval()

    print(f'[i] architecture: {args.architecture}')
    print(f'[i] model_type: {args.model_type}')
    print(f'[i] backbone: {args.backbone}')
    print(f'[i] total params: {calculate_parameters(model):.2f}M')
    print()

    load_model(model, model_path, parallel=False)

    scales = [float(scale) for scale in args.scales.split(',')]
    eval_timer = Timer()
    eval_timer.tik()

    with torch.no_grad():
        length = len(dataset)

        for step, (ori_image, image_id, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size
            logits_list = []

            for scale in scales:
                scaled_image = copy.deepcopy(ori_image)
                scaled_image = scaled_image.resize(
                    (round(ori_w * scale), round(ori_h * scale)),
                    resample=PIL.Image.CUBIC
                )

                scaled_image = normalize_fn(scaled_image)
                scaled_image = scaled_image.transpose((2, 0, 1))
                scaled_image = torch.from_numpy(scaled_image)

                flipped_image = scaled_image.flip(-1)
                images = torch.stack([scaled_image, flipped_image])

                logits = inference(model, images, (ori_h, ori_w), args.model_type)
                logits_list.append(logits)

            preds = np.sum(logits_list, axis=0)
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()

            if args.iteration > 0:
                preds = crf_inference(
                    np.asarray(ori_image),
                    preds.transpose((2, 0, 1)),
                    t=args.iteration
                )
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)

            save_path = os.path.join(pred_dir, f'{image_id}.png')
            imageio.imwrite(save_path, pred_mask.astype(np.uint8))

            sys.stdout.write(
                '\r# Inference [{}/{}] = {:.2f}%'.format(
                    step + 1, length, (step + 1) / length * 100
                )
            )
            sys.stdout.flush()

        print()
        print('inference time:', eval_timer.tok(clear=True))

    if args.domain == 'val':
        print(
            "python3 evaluate.py --experiment_name {} --domain {} --mode png".format(
                experiment_tag, args.domain
            )
        )


if __name__ == '__main__':
    main()