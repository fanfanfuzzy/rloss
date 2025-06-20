#!/usr/bin/env python3
import torch
import numpy as np
import sys
import os

def test_pytorch_gpu():
    print("=== PyTorch GPU Environment Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available - CPU only mode")
    
    print(f"NumPy version: {np.__version__}")
    return torch.cuda.is_available()

def test_bilateral_filter():
    print("\n=== Bilateral Filter Extension Test ===")
    try:
        sys.path.append("wrapper/bilateralfilter/")
        from bilateralfilter import bilateralfilter, bilateralfilter_batch
        print("✓ Bilateral filter import successful")
        return True
    except ImportError as e:
        print(f"✗ Bilateral filter import failed: {e}")
        return False

def test_densecrf_loss():
    print("\n=== DenseCRF Loss Test ===")
    try:
        sys.path.append("pytorch-deeplab_v3_plus/")
        from DenseCRFLoss import DenseCRFLoss
        
        losslayer = DenseCRFLoss(weight=1e-9, sigma_rgb=15.0, sigma_xy=80.0, scale_factor=0.5)
        print("✓ DenseCRF loss layer creation successful")
        print(f"✓ Loss layer config: {losslayer}")
        return True
    except Exception as e:
        print(f"✗ DenseCRF loss test failed: {e}")
        return False

def test_basic_training_imports():
    print("\n=== Training Script Imports Test ===")
    try:
        sys.path.append("pytorch-deeplab_v3_plus/")
        from mypath import Path
        from dataloaders import make_data_loader
        from modeling.deeplab import DeepLab
        print("✓ Core training imports successful")
        
        path = Path.db_root_dir('pascal')
        print(f"✓ Dataset path configured: {path}")
        return True
    except Exception as e:
        print(f"✗ Training imports test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing rloss environment setup...")
    
    gpu_ok = test_pytorch_gpu()
    bilateral_ok = test_bilateral_filter()
    densecrf_ok = test_densecrf_loss()
    training_ok = test_basic_training_imports()
    
    print("\n=== Test Summary ===")
    print(f"PyTorch GPU: {'✓' if gpu_ok else '✗'}")
    print(f"Bilateral Filter: {'✓' if bilateral_ok else '✗'}")
    print(f"DenseCRF Loss: {'✓' if densecrf_ok else '✗'}")
    print(f"Training Imports: {'✓' if training_ok else '✗'}")
    
    if all([bilateral_ok, densecrf_ok, training_ok]):
        print("\n🎉 Environment setup successful!")
        exit(0)
    else:
        print("\n❌ Environment setup incomplete - see errors above")
        exit(1)
