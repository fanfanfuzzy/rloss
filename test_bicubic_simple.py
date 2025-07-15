#!/usr/bin/env python3
"""
Simple test for bicubic interpolation functionality
"""
import sys
import os
sys.path.append('pytorch/pytorch-deeplab_v3_plus')

def test_bicubic_import():
    """Test that bicubic transformation can be imported and used"""
    try:
        from dataloaders import custom_transforms as tr
        from PIL import Image
        print("✅ Successfully imported custom_transforms")
        
        transform = tr.FixScaleCropImageBicubic(crop_size=64, interpolation='bicubic')
        print("✅ FixScaleCropImageBicubic class created successfully")
        
        test_img = Image.new('RGB', (128, 128), color='red')
        result = transform(test_img)
        print(f"✅ Bicubic transformation successful: input {test_img.size} -> output {result.size}")
        
        transform_bilinear = tr.FixScaleCropImageBicubic(crop_size=64, interpolation='bilinear')
        result_bilinear = transform_bilinear(test_img)
        print(f"✅ Bilinear transformation successful: input {test_img.size} -> output {result_bilinear.size}")
        
        print("🎉 All bicubic interpolation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_bicubic_import()
    sys.exit(0 if success else 1)
