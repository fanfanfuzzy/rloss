#!/usr/bin/env python3
"""
小画像（40x40など）専用の推論スクリプト
rloss framework for small image inference with optimized parameters
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from os.path import join, isdir
import sys
import torch
import torch.nn as nn

sys.path.append('../pytorch/pytorch-deeplab_v3_plus')
from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from dataloaders.utils import decode_segmap
from dataloaders import custom_transforms as tr
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.saver import Saver
import time
import multiprocessing

from DenseCRFLoss import DenseCRFLoss

def get_optimal_params(image_size):
    """画像サイズに基づいて最適なパラメータを返す"""
    if image_size <= 64:
        return {
            'crop_size': 64,
            'rloss_scale': 0.25,
            'sigma_xy': 20,
            'sigma_rgb': 10,
            'batch_size': 32
        }
    elif image_size <= 128:
        return {
            'crop_size': 128,
            'rloss_scale': 0.5,
            'sigma_xy': 40,
            'sigma_rgb': 15,
            'batch_size': 16
        }
    elif image_size <= 256:
        return {
            'crop_size': 256,
            'rloss_scale': 0.75,
            'sigma_xy': 60,
            'sigma_rgb': 15,
            'batch_size': 8
        }
    else:
        return {
            'crop_size': 513,
            'rloss_scale': 1.0,
            'sigma_xy': 80,
            'sigma_rgb': 15,
            'batch_size': 4
        }

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Small Image Inference")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: mobilenet)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--n_class', type=int, default=21)
    parser.add_argument('--crop_size', type=int, default=None,
                        help='crop image size (auto-detected if not specified)')
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    
    # checking point
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint file')
    
    parser.add_argument('--rloss-weight', type=float, default=2e-9,
                        metavar='M', help='densecrf loss weight')
    parser.add_argument('--rloss-scale', type=float, default=None,
                        help='scale factor for rloss input (auto-detected if not specified)')
    parser.add_argument('--sigma-rgb', type=float, default=None,
                        help='DenseCRF sigma_rgb (auto-detected if not specified)')
    parser.add_argument('--sigma-xy', type=float, default=None,
                        help='DenseCRF sigma_xy (auto-detected if not specified)')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='input image path')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='output directory')
    
    parser.add_argument('--batch_process', action='store_true',
                        help='process all images in input directory')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if os.path.isfile(args.image_path):
        sample_img = Image.open(args.image_path)
        img_size = min(sample_img.size)
        print(f"Detected image size: {sample_img.size}, min dimension: {img_size}")
    else:
        img_size = 64  # デフォルト
        print("Using default parameters for small images")
    
    optimal_params = get_optimal_params(img_size)
    
    if args.crop_size is None:
        args.crop_size = optimal_params['crop_size']
    if args.rloss_scale is None:
        args.rloss_scale = optimal_params['rloss_scale']
    if args.sigma_rgb is None:
        args.sigma_rgb = optimal_params['sigma_rgb']
    if args.sigma_xy is None:
        args.sigma_xy = optimal_params['sigma_xy']
    
    print(f"Using optimized parameters:")
    print(f"  crop_size: {args.crop_size}")
    print(f"  rloss_scale: {args.rloss_scale}")
    print(f"  sigma_rgb: {args.sigma_rgb}")
    print(f"  sigma_xy: {args.sigma_xy}")
    
    # Define network
    model = DeepLab(num_classes=args.n_class,
                    backbone=args.backbone,
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)
    
    # Using cuda
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
    
    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()
    
    if not isdir(args.output_directory):
        os.makedirs(args.output_directory)
    
    if args.batch_process and os.path.isdir(args.image_path):
        image_files = [f for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(args.image_path, img_file)
            process_single_image(model, img_path, args)
    else:
        process_single_image(model, args.image_path, args)

def process_single_image(model, image_path, args):
    """単一画像の処理"""
    composed_transforms = transforms.Compose([
        tr.FixScaleCropImageBicubic(crop_size=args.crop_size, interpolation='bicubic'),
        tr.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensorImage()])
    
    image = composed_transforms(Image.open(image_path).convert('RGB')).unsqueeze(0)
    image_cpu = image
    if args.cuda:
        image = image.cuda()
    
    start = time.time()
    with torch.no_grad():
        output = model(image)
    inference_time = time.time() - start
    print(f'Inference time: {inference_time:.4f}s')
    
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    
    segmap = decode_segmap(pred[0], 'pascal') * 255
    segmap = segmap.astype(np.uint8)
    segimg = Image.fromarray(segmap, 'RGB')
    
    base_name = os.path.basename(image_path).split('.')[0]
    segimg.save(join(args.output_directory, f'{base_name}_prediction.png'))
    
    if args.rloss_weight > 0:
        softmax = nn.Softmax(dim=1)
        probs = softmax(output)
        probs = Variable(probs, requires_grad=True)
        
        croppings = torch.ones(pred.shape).float()
        if args.cuda:
            croppings = croppings.cuda()
        
        densecrflosslayer = DenseCRFLoss(
            weight=args.rloss_weight, 
            sigma_rgb=args.sigma_rgb, 
            sigma_xy=args.sigma_xy, 
            scale_factor=args.rloss_scale
        )
        if args.cuda:
            densecrflosslayer.cuda()
        
        densecrfloss = densecrflosslayer(image_cpu, probs, croppings)
        print(f"DenseCRF loss: {densecrfloss.item():.6f}")
        
        densecrfloss.backward()
        grad_seg = probs.grad.cpu().numpy()
        
        for i in range(min(5, args.n_class)):  # 最初の5クラスのみ
            if np.max(np.abs(grad_seg[0, i, :, :])) > 1e-6:  # 有意な勾配がある場合のみ
                fig = plt.figure(figsize=(8, 6))
                plt.imshow(grad_seg[0, i, :, :], cmap="hot")
                plt.colorbar()
                plt.title(f'DenseCRF Gradient - Class {i}')
                plt.axis('off')
                plt.savefig(join(args.output_directory, f'{base_name}_grad_class_{i}.png'))
                plt.close(fig)

if __name__ == "__main__":
    main()
