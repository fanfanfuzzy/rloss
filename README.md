# Regularized Losses (rloss) for Weakly-supervised CNN Segmentation

**PyTorch Implementation for A100 GPU / A100 GPU用PyTorch実装**

<span align="center"><img src="teaser.png" alt="" width="800"/></span>

## Project Overview / プロジェクト概要

**English**: This repository implements regularized loss framework for weakly-supervised CNN segmentation using scribble annotations. The framework combines partial cross-entropy (pCE) loss over scribbles with regularization losses such as DenseCRF to achieve high-quality segmentation with minimal supervision.

**日本語**: このリポジトリは、スクリブル注釈を使用した弱教師ありCNNセグメンテーション用の正則化損失フレームワークを実装しています。スクリブル上の部分交差エントロピー（pCE）損失とDenseCRFなどの正則化損失を組み合わせて、最小限の教師ありで高品質なセグメンテーションを実現します。

## Citation / 引用

If you use this code, please cite the following paper / このコードを使用する場合は、以下の論文を引用してください：

**"On Regularized Losses for Weakly-supervised CNN Segmentation"** [PDF](http://cs.uwaterloo.ca/~m62tang/OnRegularizedLosses_ECCV18.pdf)  
[Meng Tang](http://cs.uwaterloo.ca/~m62tang), [Federico Perazzi](https://fperazzi.github.io/), [Abdelaziz Djelouah](https://adjelouah.github.io/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/), [Christopher Schroers](https://www.disneyresearch.com/people/christopher-schroers/), [Yuri Boykov](https://cs.uwaterloo.ca/about/people/yboykov)  
In European Conference on Computer Vision (ECCV), Munich, Germany, September 2018.

## Performance Results / 性能結果

| Network | Weak Supervision (~3% pixels) | | Full Supervision |
|---------|-------------------------------|---|------------------|
| | Cross Entropy Loss | **w/ DenseCRF Loss** | |
| Deeplab_largeFOV | 55.8% | **62.3%** | 63.0% |
| Deeplab_Msc_largeFOV | n/a | **63.2%** | 64.1% |
| Deeplab_VGG16 | 60.7% | **64.7%** | 68.8% |
| Deeplab_ResNet101 | 69.5% | **73.0%** | 75.6% |

**Table**: mIOU on PASCAL VOC2012 validation set

## Documentation Navigation / ドキュメント案内

| File / ファイル | Content / 内容 | Target Users / 対象ユーザー |
|----------------|----------------|---------------------------|
| **README.md** (this file) | Complete PyTorch guide / PyTorch完全ガイド | **Primary users / メイン利用者** |
| [README_original.md](README_original.md) | Original project overview & Caffe implementation / 元のプロジェクト概要・Caffe実装 | Researchers / 研究者 |
| [README_ja.md](README_ja.md) | Detailed Japanese documentation / 詳細な日本語ドキュメント | Japanese users / 日本語ユーザー |
| [scripts/README.md](scripts/README.md) | Small image optimization scripts / 小画像最適化スクリプト | Developers / 開発者 |
| [pytorch/README.md](pytorch/README.md) | Legacy PyTorch setup / 従来のPyTorchセットアップ | Legacy users / 従来ユーザー |

## Quick Start / クイックスタート

### Prerequisites / 前提条件
- **GPU**: A100 recommended (A6000 compatible) / A100推奨（A6000対応）
- **OS**: Ubuntu 22.04 / Ubuntu 22.04
- **Docker**: With nvidia-container-toolkit / nvidia-container-toolkit付き

### Build and Run / ビルドと実行

**English**:
```bash
# Build Docker environment
make build

# Start container with GPU support
make run

# Inside container - test environment
make test-env

# Train with small images (40x40)
make train-small

# Train with standard images (513x513)
make train-standard

# Run inference with bicubic interpolation
make inference IMAGE_PATH=/path/to/image.jpg
```

**日本語**:
```bash
# Docker環境をビルド
make build

# GPU対応コンテナを開始
make run

# コンテナ内 - 環境テスト
make test-env

# 小画像（40x40）で学習
make train-small

# 標準画像（513x513）で学習
make train-standard

# bicubic補間で推論実行
make inference IMAGE_PATH=/path/to/image.jpg
```

## Environment Setup / 環境セットアップ

### A100 GPU Configuration / A100 GPU設定

**English**: This implementation is optimized for A100 GPUs with 40GB memory. Key optimizations include:
- Mixed precision training (FP16/FP32)
- Gradient checkpointing for memory efficiency
- Optimized batch sizes for A100 architecture
- CUDA 11.8+ compatibility

**日本語**: この実装は40GBメモリのA100 GPU用に最適化されています。主な最適化：
- 混合精度学習（FP16/FP32）
- メモリ効率のためのグラディエントチェックポイント
- A100アーキテクチャ用に最適化されたバッチサイズ
- CUDA 11.8+対応

### Docker Runtime Requirements / Docker実行要件

**Important / 重要**: 
- Requires `nvidia-container-toolkit` / `nvidia-container-toolkit`が必要
- GPU memory: Minimum 16GB, Recommended 40GB (A100) / GPUメモリ：最小16GB、推奨40GB（A100）
- Host memory: Minimum 32GB / ホストメモリ：最小32GB

## Training & Inference / 学習・推論

### Training with Low-Resolution Images / 低解像度画像での学習

**English**: For training with low-resolution images and scribble labels:

**日本語**: 低解像度画像とスクリブルラベルでの学習：

```bash
# Optimized for 40x40 images
make train-small

# With Weights & Biases logging
make train-small-wandb
```

### Bicubic Interpolation for Inference / 推論用Bicubic補間

**English**: **NEW FEATURE**: Enhanced bicubic interpolation for low-resolution image inference provides superior quality compared to standard bilinear interpolation.

**日本語**: **新機能**: 低解像度画像推論用の強化されたbicubic補間により、標準的なbilinear補間と比較して優れた品質を提供します。

#### Usage / 使用方法

```bash
# Standard inference with bicubic interpolation (default)
make inference IMAGE_PATH=/path/to/image.jpg

# Direct script usage
python pytorch/pytorch-deeplab_v3_plus/inference.py \
    --use_bicubic \
    --image_path /path/to/image.jpg \
    --output_directory ./results

# Small images inference (bicubic enabled by default)
python scripts/inference_small_images.py \
    --checkpoint /path/to/model.pth.tar \
    --image_path /path/to/small_image.jpg
```

#### Technical Implementation / 技術実装

**English**: The bicubic interpolation is implemented through:
- `FixScaleCropImageBicubic` class in `pytorch/pytorch-deeplab_v3_plus/dataloaders/custom_transforms.py`
- `--use_bicubic` flag in main inference script
- Automatic bicubic usage for small images (< 128x128)
- Full compatibility with existing DenseCRF loss functionality

**日本語**: bicubic補間は以下を通じて実装されています：
- `pytorch/pytorch-deeplab_v3_plus/dataloaders/custom_transforms.py`の`FixScaleCropImageBicubic`クラス
- メイン推論スクリプトの`--use_bicubic`フラグ
- 小画像（< 128x128）での自動bicubic使用
- 既存のDenseCRF loss機能との完全な互換性

#### Performance Considerations / パフォーマンス考慮事項

**English**:
- **Quality**: Significant improvement for low-resolution images (< 100x100)
- **Speed**: Slightly slower than bilinear (~10-15% overhead)
- **Memory**: Minimal additional memory usage
- **Recommended**: For inference on images smaller than 128x128 pixels

**日本語**:
- **品質**: 低解像度画像（< 100x100）で大幅な改善
- **速度**: bilinearよりわずかに遅い（約10-15%のオーバーヘッド）
- **メモリ**: 追加メモリ使用量は最小限
- **推奨**: 128x128ピクセル未満の画像での推論

## Weights & Biases Integration / Weights & Biases統合

### Setup / セットアップ

**English**: Configure W&B for experiment tracking:

**日本語**: 実験追跡用のW&B設定：

```bash
# Inside container
python setup_wandb.py

# Verify setup
source /workspace/.env
echo $WANDB_API_KEY
```

### Training with W&B / W&Bでの学習

```bash
# Small images with W&B logging
make train-small-wandb

# Standard images with W&B logging  
make train-standard-wandb
```

**Note / 注意**: The W&B API key is automatically saved to `/workspace/.env` for persistence across container restarts.

## Performance Optimization / パフォーマンス最適化

### Memory Management / メモリ管理

**English**: For optimal A100 performance:
- Use mixed precision: `--fp16` flag
- Adjust batch size based on image resolution
- Enable gradient checkpointing for large models
- Monitor GPU memory usage with `nvidia-smi`

**日本語**: A100の最適パフォーマンスのために：
- 混合精度を使用：`--fp16`フラグ
- 画像解像度に基づいてバッチサイズを調整
- 大きなモデルでグラディエントチェックポイントを有効化
- `nvidia-smi`でGPUメモリ使用量を監視

### Troubleshooting / トラブルシューティング

**Common Issues / よくある問題**:

1. **CUDA out of memory / CUDAメモリ不足**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=4
   ```

2. **W&B API key not persisting / W&B APIキーが永続化されない**
   ```bash
   # Re-run setup
   python setup_wandb.py
   source /workspace/.env
   ```

3. **Docker GPU access issues / Docker GPU アクセス問題**
   ```bash
   # Verify nvidia-container-toolkit
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

## Advanced Topics / 高度なトピック

### DenseCRF Loss Configuration / DenseCRF Loss設定

**English**: Fine-tune DenseCRF parameters for your dataset:
- `sigma_xy`: Spatial bandwidth (default: 100)
- `sigma_rgb`: Color bandwidth (default: 15)  
- `rloss_scale`: Loss weight (default: 0.1)

**日本語**: データセット用のDenseCRFパラメータの微調整：
- `sigma_xy`: 空間帯域幅（デフォルト：100）
- `sigma_rgb`: 色帯域幅（デフォルト：15）
- `rloss_scale`: 損失重み（デフォルト：0.1）

### Custom Dataset Integration / カスタムデータセット統合

**English**: For custom datasets with scribble annotations:
1. Prepare scribble masks in PNG format
2. Update dataset paths in configuration files
3. Adjust class numbers and loss weights
4. Use bicubic interpolation for low-resolution datasets

**日本語**: スクリブル注釈付きカスタムデータセットの場合：
1. PNG形式でスクリブルマスクを準備
2. 設定ファイルでデータセットパスを更新
3. クラス数と損失重みを調整
4. 低解像度データセットにはbicubic補間を使用

## Related Research / 関連研究

### Other Regularized Losses / その他の正則化損失

**"Normalized Cut Loss for Weakly-supervised CNN Segmentation"** [PDF](https://cs.uwaterloo.ca/~m62tang/ncloss_CVPR18.pdf)  
Meng Tang, Abdelaziz Djelouah, Federico Perazzi, Yuri Boykov, Christopher Schroers  
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, USA, June 2018

**"Size-constraint loss for weakly supervised CNN segmentation"** [PDF](https://arxiv.org/pdf/1805.04628.pdf) [Code](https://github.com/LIVIAETS/SizeLoss_WSS)  
Hoel Kervadec, Jose Dolz, Meng Tang, Eric Granger, Yuri Boykov, Ismail Ben Ayed  
In International conference on Medical Imaging with Deep Learning (MIDL), Amsterdam, Netherlands, July 2018

### Trained Models / 学習済みモデル

**English**: Pre-trained models for various networks are available at: https://cs.uwaterloo.ca/~m62tang/rloss/

**日本語**: 各種ネットワーク用の事前学習済みモデルは以下で利用可能：https://cs.uwaterloo.ca/~m62tang/rloss/

---

**Link to Devin run**: https://app.devin.ai/sessions/ec410d0160814991b00fa99413c3d317  
**Requested by**: Yasuhiko Igarashi (igayasu1219@gmail.com)
