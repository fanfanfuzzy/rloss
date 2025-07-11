#!/usr/bin/env python3
import os
import shutil

def create_segmentation_aug_files():
    """
    Create SegmentationAug directory and train.txt/val.txt files if they don't exist.
    This is needed because the standard PASCAL VOC2012 dataset doesn't include SegmentationAug.
    """
    base_dir = '/data/datasets/VOCdevkit/VOC2012'
    segmentation_dir = os.path.join(base_dir, 'ImageSets', 'Segmentation')
    segmentation_aug_dir = os.path.join(base_dir, 'ImageSets', 'SegmentationAug')
    
    print(f"🔍 Checking dataset at {base_dir}")
    if not os.path.exists(base_dir):
        print(f"❌ Dataset not found at {base_dir}")
        return False
    
    print(f"🔍 Checking Segmentation directory at {segmentation_dir}")
    if not os.path.exists(segmentation_dir):
        print(f"❌ Segmentation directory not found at {segmentation_dir}")
        return False
    
    print(f"📁 Creating SegmentationAug directory...")
    os.makedirs(segmentation_aug_dir, exist_ok=True)
    print(f"✅ Created SegmentationAug directory at {segmentation_aug_dir}")
    
    train_file = os.path.join(segmentation_dir, 'train.txt')
    val_file = os.path.join(segmentation_dir, 'val.txt')
    
    train_aug_file = os.path.join(segmentation_aug_dir, 'train.txt')
    val_aug_file = os.path.join(segmentation_aug_dir, 'val.txt')
    
    print(f"📋 Copying train.txt file...")
    if os.path.exists(train_file):
        shutil.copy2(train_file, train_aug_file)
        print(f"✅ Created {train_aug_file} ({os.path.getsize(train_aug_file)} bytes)")
    else:
        print(f"❌ Source train.txt not found at {train_file}")
        return False
    
    print(f"📋 Copying val.txt file...")
    if os.path.exists(val_file):
        shutil.copy2(val_file, val_aug_file)
        print(f"✅ Created {val_aug_file} ({os.path.getsize(val_aug_file)} bytes)")
    else:
        print(f"❌ Source val.txt not found at {val_file}")
        return False
    
    print("✅ SegmentationAug files setup complete")
    print(f"📊 Final verification:")
    print(f"   - SegmentationAug directory: {os.path.exists(segmentation_aug_dir)}")
    print(f"   - train.txt exists: {os.path.exists(train_aug_file)}")
    print(f"   - val.txt exists: {os.path.exists(val_aug_file)}")
    return True

if __name__ == "__main__":
    create_segmentation_aug_files()
