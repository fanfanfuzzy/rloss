# Bicubic Interpolation Implementation Guide / Bicubic補間実装ガイド

## Overview / 概要

**English**: This guide explains the bicubic interpolation implementation for low-resolution image inference in the rloss framework. The implementation adds high-quality image upsampling capabilities while maintaining full compatibility with existing DenseCRF loss functionality.

**日本語**: このガイドでは、rlossフレームワークにおける低解像度画像推論用のbicubic補間実装について説明します。既存のDenseCRF loss機能との完全な互換性を保ちながら、高品質な画像アップサンプリング機能を追加します。

## Key Files Modified / 主要な変更ファイル

### 1. Core Transformation Class / コア変換クラス
**File**: `pytorch/pytorch-deeplab_v3_plus/dataloaders/custom_transforms.py`

**English**: Added `FixScaleCropImageBicubic` class that provides configurable interpolation (bicubic or bilinear) for image resizing during preprocessing.

**日本語**: 画像前処理時のリサイズに設定可能な補間（bicubicまたはbilinear）を提供する`FixScaleCropImageBicubic`クラスを追加。

```python
# New class added / 新しく追加されたクラス
class FixScaleCropImageBicubic(object):
    def __init__(self, crop_size, interpolation='bicubic'):
        self.crop_size = crop_size
        self.interpolation = Resampling.BICUBIC if interpolation == 'bicubic' else Resampling.BILINEAR
```

### 2. Main Inference Script / メイン推論スクリプト
**File**: `pytorch/pytorch-deeplab_v3_plus/inference.py`

**English**: Added `--use_bicubic` command-line flag to enable bicubic interpolation during inference.

**日本語**: 推論時にbicubic補間を有効にする`--use_bicubic`コマンドラインフラグを追加。

```bash
# Usage example / 使用例
python inference.py --use_bicubic --image_path /path/to/image.jpg
```

### 3. Small Images Inference Script / 小画像推論スクリプト
**File**: `scripts/inference_small_images.py`

**English**: Updated to use bicubic interpolation by default for low-resolution images (40x40 optimized).

**日本語**: 低解像度画像（40x40最適化）に対してデフォルトでbicubic補間を使用するように更新。

### 4. Build Configuration / ビルド設定
**File**: `pytorch/Makefile`

**English**: Updated inference targets to include `--use_bicubic` flag by default.

**日本語**: 推論ターゲットにデフォルトで`--use_bicubic`フラグを含むように更新。

## Usage Instructions / 使用方法

### For Standard Inference / 標準推論の場合

**English**:
```bash
# With bicubic interpolation (recommended for low-res images)
make inference IMAGE_PATH=/path/to/image.jpg

# Or directly
python pytorch/pytorch-deeplab_v3_plus/inference.py \
    --use_bicubic \
    --image_path /path/to/image.jpg \
    --output_directory ./results
```

**日本語**:
```bash
# bicubic補間使用（低解像度画像に推奨）
make inference IMAGE_PATH=/path/to/image.jpg

# または直接実行
python pytorch/pytorch-deeplab_v3_plus/inference.py \
    --use_bicubic \
    --image_path /path/to/image.jpg \
    --output_directory ./results
```

### For Small Images Inference / 小画像推論の場合

**English**:
```bash
# Automatically uses bicubic interpolation
python scripts/inference_small_images.py \
    --checkpoint /path/to/model.pth.tar \
    --image_path /path/to/small_image.jpg
```

**日本語**:
```bash
# 自動的にbicubic補間を使用
python scripts/inference_small_images.py \
    --checkpoint /path/to/model.pth.tar \
    --image_path /path/to/small_image.jpg
```

## Technical Details / 技術詳細

### Interpolation Methods / 補間方法

**English**:
- **Bicubic**: Higher quality, smoother results for upsampling low-resolution images
- **Bilinear**: Faster processing, maintained for backward compatibility
- **Default**: Bicubic for small images, configurable for standard inference

**日本語**:
- **Bicubic**: 低解像度画像のアップサンプリングでより高品質で滑らかな結果
- **Bilinear**: より高速な処理、後方互換性のために維持
- **デフォルト**: 小画像にはbicubic、標準推論では設定可能

### Integration with DenseCRF / DenseCRFとの統合

**English**: The bicubic interpolation is fully compatible with the existing DenseCRF loss framework. No changes are required to DenseCRF parameters (`sigma_xy`, `sigma_rgb`, `rloss_scale`).

**日本語**: bicubic補間は既存のDenseCRF lossフレームワークと完全に互換性があります。DenseCRFパラメータ（`sigma_xy`、`sigma_rgb`、`rloss_scale`）の変更は不要です。

### Performance Considerations / パフォーマンス考慮事項

**English**:
- Bicubic interpolation has slightly higher computational cost than bilinear
- Quality improvement is most noticeable for low-resolution images (< 100x100)
- Recommended for inference on images smaller than 128x128 pixels

**日本語**:
- bicubic補間はbilinearよりもわずかに計算コストが高い
- 品質向上は低解像度画像（< 100x100）で最も顕著
- 128x128ピクセル未満の画像での推論に推奨

## Testing / テスト

**English**: Two test scripts are provided to verify the implementation:

**日本語**: 実装を検証するために2つのテストスクリプトが提供されています：

```bash
# Basic functionality test / 基本機能テスト
python test_bicubic_simple.py

# Visual comparison test / 視覚的比較テスト
python test_bicubic_inference.py /path/to/test_image.jpg
```

## Backward Compatibility / 後方互換性

**English**: All existing functionality is preserved. The original `FixScaleCropImage` class remains unchanged, and bicubic interpolation is opt-in for standard inference.

**日本語**: 既存の機能はすべて保持されています。元の`FixScaleCropImage`クラスは変更されておらず、標準推論ではbicubic補間はオプトインです。

## Next Steps for Training / 学習の次のステップ

**English**: To train with low-resolution images and scribble labels:

**日本語**: 低解像度画像とスクリブルラベルで学習するには：

1. **Prepare dataset / データセット準備**: Ensure scribble annotations are available
2. **Training configuration / 学習設定**: Use `make train-small` for 40x40 images
3. **Inference / 推論**: Use the bicubic-enabled inference scripts for better quality

## Support / サポート

**English**: For issues or questions about the bicubic interpolation implementation, refer to the pull request: https://github.com/fanfanfuzzy/rloss/pull/17

**日本語**: bicubic補間実装に関する問題や質問については、プルリクエストを参照してください：https://github.com/fanfanfuzzy/rloss/pull/17
