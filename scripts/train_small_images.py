#!/usr/bin/env python3
"""
小画像データセット用の訓練スクリプト
Optimized training script for small image datasets with rloss framework
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('../pytorch/pytorch-deeplab_v3_plus')
from train_withdensecrfloss import main as original_main

def get_optimal_training_params(image_size):
    """画像サイズに基づいて最適な訓練パラメータを返す"""
    if image_size <= 64:
        return {
            'crop_size': 64,
            'base_size': 64,
            'batch_size': 32,
            'lr': 0.01,
            'epochs': 80,
            'rloss_scale': 0.25,
            'sigma_xy': 20,
            'sigma_rgb': 10
        }
    elif image_size <= 128:
        return {
            'crop_size': 128,
            'base_size': 128,
            'batch_size': 16,
            'lr': 0.007,
            'epochs': 60,
            'rloss_scale': 0.5,
            'sigma_xy': 40,
            'sigma_rgb': 15
        }
    elif image_size <= 256:
        return {
            'crop_size': 256,
            'base_size': 256,
            'batch_size': 8,
            'lr': 0.007,
            'epochs': 60,
            'rloss_scale': 0.75,
            'sigma_xy': 60,
            'sigma_rgb': 15
        }
    else:
        return {
            'crop_size': 513,
            'base_size': 513,
            'batch_size': 4,
            'lr': 0.007,
            'epochs': 60,
            'rloss_scale': 1.0,
            'sigma_xy': 80,
            'sigma_rgb': 15
        }

def create_optimized_args(image_size, dataset_name, checkpoint_path=None):
    """最適化された引数リストを作成"""
    params = get_optimal_training_params(image_size)
    
    args = [
        '--backbone', 'mobilenet',
        '--dataset', dataset_name,
        '--crop-size', str(params['crop_size']),
        '--base-size', str(params['base_size']),
        '--batch-size', str(params['batch_size']),
        '--lr', str(params['lr']),
        '--epochs', str(params['epochs']),
        '--densecrfloss', '2e-9',
        '--rloss-scale', str(params['rloss_scale']),
        '--sigma-rgb', str(params['sigma_rgb']),
        '--sigma-xy', str(params['sigma_xy']),
        '--workers', '4',
        '--eval-interval', '2',
        '--save-interval', '5',
        '--checkname', f'deeplab-{dataset_name}-size{image_size}'
    ]
    
    if checkpoint_path:
        args.extend(['--resume', checkpoint_path, '--ft'])
    
    return args

def main():
    parser = argparse.ArgumentParser(description="Optimized Training for Small Images")
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name')
    parser.add_argument('--image_size', type=int, required=True,
                        help='typical image size in your dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to pretrained checkpoint for fine-tuning')
    parser.add_argument('--custom_params', action='store_true',
                        help='allow manual parameter override')
    
    parser.add_argument('--custom_crop_size', type=int, default=None)
    parser.add_argument('--custom_batch_size', type=int, default=None)
    parser.add_argument('--custom_lr', type=float, default=None)
    parser.add_argument('--custom_epochs', type=int, default=None)
    
    args = parser.parse_args()
    
    print(f"=== 小画像データセット用最適化訓練 ===")
    print(f"データセット: {args.dataset}")
    print(f"画像サイズ: {args.image_size}x{args.image_size}")
    
    optimized_args = create_optimized_args(
        args.image_size, 
        args.dataset, 
        args.checkpoint
    )
    
    if args.custom_params:
        if args.custom_crop_size:
            idx = optimized_args.index('--crop-size')
            optimized_args[idx + 1] = str(args.custom_crop_size)
        if args.custom_batch_size:
            idx = optimized_args.index('--batch-size')
            optimized_args[idx + 1] = str(args.custom_batch_size)
        if args.custom_lr:
            idx = optimized_args.index('--lr')
            optimized_args[idx + 1] = str(args.custom_lr)
        if args.custom_epochs:
            idx = optimized_args.index('--epochs')
            optimized_args[idx + 1] = str(args.custom_epochs)
    
    print(f"最適化されたパラメータ:")
    for i in range(0, len(optimized_args), 2):
        if i + 1 < len(optimized_args):
            print(f"  {optimized_args[i]}: {optimized_args[i + 1]}")
    
    print(f"\n訓練を開始します...")
    sys.argv = ['train_withdensecrfloss.py'] + optimized_args
    
    try:
        original_main()
    except Exception as e:
        print(f"訓練中にエラーが発生しました: {e}")
        print("パラメータを調整して再試行してください。")
        return 1
    
    print("訓練が完了しました！")
    return 0

if __name__ == "__main__":
    exit(main())
