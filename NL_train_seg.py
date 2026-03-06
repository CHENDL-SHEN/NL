# Copyright (C) 2020 * Ltd. All rights reserved.
# author: Sanghyeon Jo <josanghyeokn@gmail.com>

"""
Training script for semantic segmentation.

Supported training modes:
    1) CE
    2) CE + PNS
    3) CE + NLMS
    4) CE + alpha * PNS + beta * NLMS

In this version, all losses are computed on the full pseudo-label batch.
If your PNS loss function later requires a different input format,
you can further modify only that part.
"""

import os
import sys
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.nlmsloss_utils import *
from tools.general.pnsloss_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--data_dir', default='/media/ders/XS/dataset/VOC2012/', type=str)
    parser.add_argument('--domain', default='T-101-hard-id-0.8', type=str)

    # Network
    parser.add_argument('--architecture', default='DeepLabv3+', type=str)
    parser.add_argument('--backbone', default='resnest101', type=str)
    parser.add_argument('--mode', default='fix', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)

    # Optimization
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--lr', default=0.007, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    # Image preprocessing
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=256, type=int)
    parser.add_argument('--max_image_size', default=1024, type=int)

    # Logging
    parser.add_argument('--print_ratio', default=1.0, type=float)
    parser.add_argument('--tag', default='TEST', type=str)

    # Pseudo labels
    parser.add_argument('--label_name', default='pcam', type=str)

    # Loss type
    parser.add_argument(
        '--use_loss',
        default='CE_PNR_MS',
        type=str,
        choices=['CE', 'CE_PNR', 'CE_MS', 'CE_PNR_MS']
    )
    parser.add_argument('--use_NonLocal', default=True, type=str2bool)

    # Loss weights
    parser.add_argument('--sigma', default=7.0, type=float)
    parser.add_argument('--alpha_pnr', default=0.1, type=float)
    parser.add_argument('--beta_nlms', default=1.0, type=float)

    return parser


def get_dataset(args, pred_dir, domain, pselabel):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])

    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])

    train_dataset = VOC_Dataset_For_WSSS(
    args.data_dir, domain, pred_dir, train_transform
    )

    valid_dataset = VOC_Dataset_For_Segmentation(
        args.data_dir, 'val', test_transform
    )

    return train_dataset, valid_dataset, train_transform


def evaluate(loader, model, eval_timer, writer, train_dataset, iteration):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    model.eval()
    eval_timer.tik()

    meter = Calculator_For_mIoU('./data/VOC_2012.json')

    with torch.no_grad():
        length = len(loader)

        for step, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits, _, _, _ = model(images)
            predictions = torch.argmax(logits, dim=1)

            if step == 0:
                vis_num = min(8, images.size(0))
                for b in range(vis_num):
                    image = get_numpy_from_tensor(images[b])
                    pred_mask = get_numpy_from_tensor(predictions[b])

                    image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                    h, w, _ = image.shape

                    pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    image = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)[..., ::-1]
                    image = image.astype(np.float32) / 255.0

                    writer.add_image(
                        f'Mask/{b + 1}',
                        image,
                        iteration,
                        dataformats='HWC'
                    )

            for batch_index in range(images.size(0)):
                pred_mask = get_numpy_from_tensor(predictions[batch_index])
                gt_mask = get_numpy_from_tensor(labels[batch_index])

                h, w = pred_mask.shape
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                meter.add(pred_mask, gt_mask)

            sys.stdout.write(
                '\r# Evaluation [{}/{}] = {:.2f}%'.format(
                    step + 1, length, (step + 1) / length * 100
                )
            )
            sys.stdout.flush()

    print()
    model.train()
    return meter.get(clear=True)


def main():
    args = get_argparser().parse_args()

    # Reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tag = args.tag
    domain = args.domain
    max_epoch = args.max_epoch
    pselabel = args.label_name

    log_dir = create_directory('./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{tag}/')
    pred_dir = f'./predata/pseudo-labels/{pselabel}/'

    log_path = os.path.join(log_dir, f'{tag}.txt')
    model_path = os.path.join(model_dir, f'{tag}.pth')

    log_func = lambda string='': log_print(string, log_path)

    log_func(f'[i] {tag}')
    log_func(f'[i] pseudo labels: {pselabel}')
    log_func()

    # Dataset and DataLoader
    train_dataset, valid_dataset, train_transform = get_dataset(args, pred_dir, domain, pselabel)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    meta_dic = read_json('./data/VOC_2012.json')

    log_func(f'[i] mean values: {imagenet_mean}')
    log_func(f'[i] std values: {imagenet_std}')
    log_func(f'[i] number of classes: {meta_dic["classes"]}')
    log_func(f'[i] train_transform: {train_transform}')
    log_func()

    val_iteration = len(train_loader)
    log_iteration = max(1, int(val_iteration * args.print_ratio))
    max_iteration = max_epoch * val_iteration

    log_func(f'[i] log_iteration: {log_iteration:,}')
    log_func(f'[i] val_iteration: {val_iteration:,}')
    log_func(f'[i] max_iteration: {max_iteration:,}')
    log_func()

    # Network
    model = DeepLabv3_Plus_PMS(
        args.backbone,
        num_classes=meta_dic['classes'] + 1,
        mode=args.mode,
        use_group_norm=args.use_gn
    )

    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ]

    model = model.cuda()
    model.train()

    log_func(f'[i] architecture: {args.architecture}')
    log_func(f'[i] backbone: {args.backbone}')
    log_func(f'[i] total params: {calculate_parameters(model):.2f}M')
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    num_gpus = len(use_gpu.split(','))
    if num_gpus > 1:
        log_func(f'[i] number of GPUs: {num_gpus}')
        model = nn.DataParallel(model)

    save_model_fn = lambda: save_model(model, model_path, parallel=num_gpus > 1)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()
    pnr_loss_fn = PNRloss(21, False)
    ms_loss_fn = MSloss()

    # Optimizer
    optimizer = PolyOptimizer(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
        max_step=max_iteration,
        nesterov=args.nesterov
    )

    train_timer = Timer()
    eval_timer = Timer()
    train_meter = Average_Meter(['loss'])

    best_valid_mIoU = -1.0
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    for iteration in range(max_iteration):
        batch = train_iterator.get()

        # Some datasets may return (images, labels, HT) or more fields.
        # Here we only use images and labels for the current training logic.
        images = batch[0].cuda(non_blocking=True)
        labels = batch[1].cuda(non_blocking=True)

        _, _, imgH, imgW = images.shape

        optimizer.zero_grad()

        logits, f_sem, f_hie, f_output = model(images)

        loss_ce = None
        loss_pnr = None
        loss_ms = None
        loss_ls = None
        loss_tv = None

        if args.use_loss == 'CE':
            loss_ce = ce_loss_fn(logits, labels)
            loss = loss_ce

            print(
                'Loss_CE = {:.6f}\tLoss = {:.6f}'.format(
                    loss_ce.item(),
                    loss.item()
                )
            )

        elif args.use_loss == 'CE_PNR':
            loss_ce = ce_loss_fn(logits, labels)
            loss_pnr = pnr_loss_fn(logits, f_sem, f_hie, imgH, imgW, labels, args.sigma)
            loss = loss_ce + args.alpha_pnr * loss_pnr

            print(
                'Loss_CE = {:.6f}\tLoss_PNR = {:.6f}\talpha = {:.6f}\tLoss = {:.6f}'.format(
                    loss_ce.item(),
                    loss_pnr.item(),
                    args.alpha_pnr,
                    loss.item()
                )
            )

        elif args.use_loss == 'CE_MS':
            probs = F.softmax(logits, dim=1)
            f_output_probs = F.softmax(f_output, dim=1)

            loss_ce = ce_loss_fn(logits, labels)
            loss_ls, loss_tv, loss_ms = ms_loss_fn(
                probs, images, f_sem, f_output_probs, args.use_NonLocal
            )
            loss = loss_ce + args.beta_nlms * loss_ms

            loss_tv_value = loss_tv.item() if torch.is_tensor(loss_tv) else float(loss_tv)

            print(
                'Loss_CE = {:.6f}\tLoss_NLMS = {:.6f}\tbeta = {:.6f}\tLoss_LS = {:.6f}\tLoss_TV = {:.6f}\tLoss = {:.6f}'.format(
                    loss_ce.item(),
                    loss_ms.item(),
                    args.beta_nlms,
                    loss_ls.item(),
                    loss_tv_value,
                    loss.item()
                )
            )

        elif args.use_loss == 'CE_PNR_MS':
            probs = F.softmax(logits, dim=1)
            f_output_probs = F.softmax(f_output, dim=1)

            loss_ce = ce_loss_fn(logits, labels)
            loss_pnr = pnr_loss_fn(logits, f_sem, f_hie, imgH, imgW, labels, args.sigma)
            loss_ls, loss_tv, loss_ms = ms_loss_fn(
                probs, images, f_sem, f_output_probs, args.use_NonLocal
            )

            loss = loss_ce + args.alpha_pnr * loss_pnr + args.beta_nlms * loss_ms

            loss_tv_value = loss_tv.item() if torch.is_tensor(loss_tv) else float(loss_tv)

            print(
                'Loss_CE = {:.6f}\tLoss_PNR = {:.6f}\tLoss_NLMS = {:.6f}\talpha = {:.6f}\tbeta = {:.6f}\tLoss_LS = {:.6f}\tLoss_TV = {:.6f}\tLoss = {:.6f}'.format(
                    loss_ce.item(),
                    loss_pnr.item(),
                    loss_ms.item(),
                    args.alpha_pnr,
                    args.beta_nlms,
                    loss_ls.item(),
                    loss_tv_value,
                    loss.item()
                )
            )

        else:
            raise ValueError(f'Unsupported loss type: {args.use_loss}')

        # Numerical stability check before backward
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() <= 0:
            print('Invalid loss detected. Skip this iteration.')
            continue

        loss.backward()
        optimizer.step()

        train_meter.add({'loss': loss.item()})

        if (iteration + 1) % log_iteration == 0:
            avg_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': avg_loss,
                'time': train_timer.tok(clear=True),
            }

            log_func(
                '[i] iteration={iteration:,}, learning_rate={learning_rate:.6f}, '
                'loss={loss:.6f}, time={time:.0f} sec'.format(**data)
            )

            writer.add_scalar('Train/loss', avg_loss, iteration + 1)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration + 1)

            if loss_ce is not None:
                writer.add_scalar('Train/loss_ce', loss_ce.item(), iteration + 1)
            if loss_pnr is not None:
                writer.add_scalar('Train/loss_pnr', loss_pnr.item(), iteration + 1)
            if loss_ms is not None:
                writer.add_scalar('Train/loss_nlms', loss_ms.item(), iteration + 1)
            if loss_ls is not None:
                writer.add_scalar('Train/loss_ls', loss_ls.item(), iteration + 1)

            if loss_tv is not None:
                loss_tv_value = loss_tv.item() if torch.is_tensor(loss_tv) else float(loss_tv)
                writer.add_scalar('Train/loss_tv', loss_tv_value, iteration + 1)

        if (iteration + 1) % val_iteration == 0:
            mIoU, _ = evaluate(valid_loader, model, eval_timer, writer, train_dataset, iteration + 1)

            if mIoU > best_valid_mIoU:
                best_valid_mIoU = mIoU
                save_model_fn()
                log_func('[i] save best model')

            data = {
                'iteration': iteration + 1,
                'mIoU': mIoU,
                'best_valid_mIoU': best_valid_mIoU,
                'time': eval_timer.tok(clear=True),
            }

            log_func(
                '[i] iteration={iteration:,}, mIoU={mIoU:.2f}%, '
                'best_valid_mIoU={best_valid_mIoU:.2f}%, time={time:.0f} sec'.format(**data)
            )

            writer.add_scalar('Evaluation/mIoU', mIoU, iteration + 1)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration + 1)

    writer.close()
    print(tag)
    print("Training finished.")


if __name__ == '__main__':
    main()