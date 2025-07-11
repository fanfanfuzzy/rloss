#!/usr/bin/env python3
import os

def create_segmentation_aug_files():
    """
    Create SegmentationAug directory and train.txt/val.txt files if they don't exist.
    This is needed because the standard PASCAL VOC2012 dataset doesn't include SegmentationAug.
    """
    base_dir = '/data/datasets/VOCdevkit/VOC2012'
    segmentation_dir = os.path.join(base_dir, 'ImageSets', 'Segmentation')
    segmentation_aug_dir = os.path.join(base_dir, 'ImageSets', 'SegmentationAug')
    
    if not os.path.exists(base_dir):
        print(f"❌ Dataset not found at {base_dir}")
        return False
    
    if not os.path.exists(segmentation_dir):
        print(f"❌ Segmentation directory not found at {segmentation_dir}")
        return False
    
    os.makedirs(segmentation_aug_dir, exist_ok=True)
    print(f"✅ Created SegmentationAug directory at {segmentation_aug_dir}")
    
    train_file = os.path.join(segmentation_dir, 'train.txt')
    val_file = os.path.join(segmentation_dir, 'val.txt')
    
    train_aug_file = os.path.join(segmentation_aug_dir, 'train.txt')
    val_aug_file = os.path.join(segmentation_aug_dir, 'val.txt')
    
    if os.path.exists(train_file) and not os.path.exists(train_aug_file):
        with open(train_file, 'r') as src, open(train_aug_file, 'w') as dst:
            dst.write(src.read())
        print(f"✅ Created {train_aug_file}")
    
    if os.path.exists(val_file) and not os.path.exists(val_aug_file):
        with open(val_file, 'r') as src, open(val_aug_file, 'w') as dst:
            dst.write(src.read())
        print(f"✅ Created {val_aug_file}")
    
    print("✅ SegmentationAug files setup complete")
    return True

if __name__ == "__main__":
    create_segmentation_aug_files()
