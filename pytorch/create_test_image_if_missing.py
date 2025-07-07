#!/usr/bin/env python3
import os
from PIL import Image

test_image_path = 'pytorch-deeplab_v3_plus/misc/test.png'

if not os.path.exists(test_image_path):
    print(f"Creating test image at {test_image_path}")
    os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
    
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img.save(test_image_path)
    print(f"✅ Test image created successfully")
else:
    print(f"✅ Test image already exists at {test_image_path}")
