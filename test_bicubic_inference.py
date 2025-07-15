#!/usr/bin/env python3
"""
Test script for bicubic interpolation in inference
"""
import sys
import os
sys.path.append('pytorch/pytorch-deeplab_v3_plus')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import custom_transforms as tr

def compare_interpolations(image_path, crop_size=64):
    """Compare bilinear vs bicubic interpolation"""
    img = Image.open(image_path).convert('RGB')
    
    transform_bilinear = tr.FixScaleCropImage(crop_size=crop_size)
    img_bilinear = transform_bilinear(img)
    
    transform_bicubic = tr.FixScaleCropImageBicubic(crop_size=crop_size, interpolation='bicubic')
    img_bicubic = transform_bicubic(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(img_bilinear)
    axes[1].set_title('Bilinear')
    axes[1].axis('off')
    axes[2].imshow(img_bicubic)
    axes[2].set_title('Bicubic')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved as interpolation_comparison.png")
    
    print(f"Original size: {img.size}")
    print(f"Processed size: {img_bilinear.size}")
    print(f"Crop size: {crop_size}")

def test_different_sizes():
    """Test bicubic interpolation with different crop sizes"""
    test_img_path = 'pytorch/pytorch-deeplab_v3_plus/misc/test.png'
    if not os.path.exists(test_img_path):
        test_img = Image.new('RGB', (256, 256), color='red')
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        test_img.save(test_img_path)
        print(f"Created test image at {test_img_path}")
    
    crop_sizes = [40, 64, 128, 256]
    for crop_size in crop_sizes:
        print(f"\nTesting crop size: {crop_size}")
        try:
            compare_interpolations(test_img_path, crop_size)
            print(f"✅ Crop size {crop_size} - Success")
        except Exception as e:
            print(f"❌ Crop size {crop_size} - Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        compare_interpolations(sys.argv[1])
    else:
        print("No image path provided, testing with different sizes...")
        test_different_sizes()
