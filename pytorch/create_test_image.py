from PIL import Image
import os

img = Image.new('RGB', (224, 224), color=(128, 128, 128))

os.makedirs('pytorch-deeplab_v3_plus/misc', exist_ok=True)
img.save('pytorch-deeplab_v3_plus/misc/test.png')
print("Test image created at pytorch-deeplab_v3_plus/misc/test.png")
