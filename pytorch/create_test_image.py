import numpy as np
from PIL import Image
import os

img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)

os.makedirs('pytorch-deeplab_v3_plus/misc', exist_ok=True)
img.save('pytorch-deeplab_v3_plus/misc/test.png')
print("Test image created at pytorch-deeplab_v3_plus/misc/test.png")
