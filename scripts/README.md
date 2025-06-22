# rloss Scripts - 小画像対応スクリプト集

このディレクトリには、小画像（40×40など）に最適化されたrlossフレームワーク用のスクリプトが含まれています。

## スクリプト一覧

### 1. inference_small_images.py
小画像専用の推論スクリプト。画像サイズを自動検出して最適なパラメータを設定します。

#### 基本的な使用方法
```bash
# 単一画像の推論
python inference_small_images.py \
    --checkpoint /path/to/model.pth.tar \
    --image_path /path/to/small_image.jpg \
    --output_directory /path/to/output/

# バッチ処理（ディレクトリ内の全画像）
python inference_small_images.py \
    --checkpoint /path/to/model.pth.tar \
    --image_path /path/to/image_directory/ \
    --output_directory /path/to/output/ \
    --batch_process
```

#### 特徴
- 画像サイズの自動検出
- 最適パラメータの自動設定
- DenseCRF損失の可視化
- バッチ処理対応

### 2. train_small_images.py
小画像データセット用の最適化された訓練スクリプト。

#### 基本的な使用方法
```bash
# 新規訓練
python train_small_images.py \
    --dataset your_small_dataset \
    --image_size 40

# ファインチューニング
python train_small_images.py \
    --dataset your_small_dataset \
    --image_size 40 \
    --checkpoint /path/to/pretrained_model.pth.tar

# カスタムパラメータ
python train_small_images.py \
    --dataset your_small_dataset \
    --image_size 40 \
    --custom_params \
    --custom_batch_size 64 \
    --custom_lr 0.02
```

#### 特徴
- 画像サイズに基づく自動パラメータ最適化
- 事前訓練済みモデルからのファインチューニング対応
- カスタムパラメータのオーバーライド機能

## 画像サイズ別推奨設定

| 画像サイズ | crop_size | batch_size | learning_rate | epochs | rloss_scale |
|-----------|-----------|------------|---------------|--------|-------------|
| 40×40     | 64        | 32         | 0.01          | 80     | 0.25        |
| 80×80     | 128       | 16         | 0.007         | 60     | 0.5         |
| 256×256   | 256       | 8          | 0.007         | 60     | 0.75        |
| 513×513   | 513       | 4          | 0.007         | 60     | 1.0         |

## 使用例

### 40×40画像での完全なワークフロー

1. **データセット準備**
```bash
# データセット構造
/data/datasets/small_dataset/
├── JPEGImages/          # 40×40 RGB画像
├── SegmentationClassAug/ # グラウンドトゥルース
└── pascal_2012_scribble/ # スクリブルアノテーション
```

2. **mypath.pyの更新**
```python
# pytorch-deeplab_v3_plus/mypath.py に追加
elif dataset == 'small_dataset':
    return '/data/datasets/small_dataset/'
```

3. **訓練実行**
```bash
python train_small_images.py \
    --dataset small_dataset \
    --image_size 40
```

4. **推論実行**
```bash
python inference_small_images.py \
    --checkpoint run/small_dataset/deeplab-small_dataset-size40/model_best.pth.tar \
    --image_path /path/to/test_images/ \
    --output_directory /path/to/results/ \
    --batch_process
```

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - バッチサイズを減らす
   - rloss_scaleを小さくする

2. **学習が進まない**
   - 学習率を調整
   - エポック数を増やす
   - 事前訓練済みモデルを使用

3. **推論結果が悪い**
   - 適切なcrop_sizeを使用
   - 訓練時と同じパラメータを使用

## 注意事項

- これらのスクリプトは元のrlossフレームワークのラッパーです
- 元のスクリプト（train_withdensecrfloss.py, inference.py）も引き続き使用可能
- 小画像に特化した最適化を提供しますが、標準サイズの画像でも使用可能

## 依存関係

- PyTorch 2.0+
- rloss framework（本リポジトリ）
- 標準的な画像処理ライブラリ（PIL, matplotlib, numpy）
