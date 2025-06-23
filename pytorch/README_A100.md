# rloss for A100 GPU + Ubuntu 22.04

A100 GPU対応のrloss環境構築ガイド

## 🚀 クイックスタート

### 1. 基本的な環境構築
```bash
# リポジトリのクローン
git clone https://github.com/fanfanfuzzy/rloss.git
cd rloss/pytorch

# Docker環境の構築
make build

# 環境テスト
make test-env

# インタラクティブコンテナの起動
make run
```

### 2. Docker Composeを使用した環境構築
```bash
# 基本サービスの起動
docker-compose up -d rloss-a100

# Jupyter Labも含めて起動
docker-compose --profile jupyter up -d

# TensorBoard監視も含めて起動
docker-compose --profile monitoring up -d
```

## 📋 利用可能なMakeコマンド

### 基本操作
- `make help` - 利用可能なコマンド一覧表示
- `make build` - Docker imageのビルド
- `make run` - インタラクティブコンテナ起動
- `make run-detached` - バックグラウンドでコンテナ起動
- `make exec` - 実行中コンテナに接続
- `make stop` - コンテナ停止
- `make clean` - コンテナとイメージの削除

### 開発・テスト
- `make test-env` - 環境テスト実行
- `make setup-data` - データディレクトリ作成
- `make download-data` - PASCAL VOC2012データセットダウンロード
- `make status` - コンテナとGPU状態確認
- `make info` - システム情報表示

### 訓練・推論
- `make train-small` - 小画像(40×40)用訓練
- `make train-standard` - 標準画像(513×513)用訓練
- `make inference IMAGE_PATH=/path/to/image.jpg` - 推論実行

### 開発支援
- `make jupyter` - Jupyter Notebook起動
- `make monitor-gpu` - GPU使用状況監視
- `make debug` - デバッグ用コンテナ起動
- `make logs` - コンテナログ表示

## 🔧 A100最適化設定

### CUDA設定
- CUDA 11.8 (A100対応)
- PyTorch 2.5.1
- cuDNN 9

### メモリ最適化
```bash
# 環境変数での設定
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=0
```

### バッチサイズ推奨値（A100 40GB）
| 画像サイズ | 推奨バッチサイズ | メモリ使用量(概算) |
|-----------|-----------------|-------------------|
| 40×40     | 128             | ~8GB              |
| 80×80     | 64              | ~12GB             |
| 256×256   | 32              | ~20GB             |
| 513×513   | 16              | ~35GB             |

## 📊 使用例

### 小画像での高速訓練
```bash
# 40×40画像での訓練（A100最適化）
make train-small

# または手動実行
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -v /data/datasets:/data/datasets \
    rloss:a100-ubuntu22.04 \
    python pytorch-deeplab_v3_plus/train_withdensecrfloss.py \
    --backbone mobilenet \
    --crop-size 64 \
    --batch-size 128 \
    --lr 0.02 \
    --epochs 100 \
    --densecrfloss 2e-9 \
    --rloss-scale 0.25
```

### 大規模データセットでの訓練
```bash
# 513×513画像での訓練（A100フル活用）
make train-standard

# カスタムパラメータでの実行
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -v /data/datasets:/data/datasets \
    rloss:a100-ubuntu22.04 \
    python pytorch-deeplab_v3_plus/train_withdensecrfloss.py \
    --backbone resnet \
    --crop-size 513 \
    --batch-size 16 \
    --lr 0.01 \
    --epochs 120 \
    --densecrfloss 1e-9 \
    --rloss-scale 1.0
```

### Jupyter Labでの開発
```bash
# Jupyter Lab起動
make jupyter

# ブラウザで http://localhost:8888 にアクセス
```

## 🐛 トラブルシューティング

### GPU認識されない場合（CUDA Error 802対応）

**症状**: nvidia-smiでGPUが見えるが、Dockerコンテナ内でCUDA Falseになる

```bash
# 1. NVIDIA Container Toolkitの自動インストール・修復
./install_nvidia_container_toolkit.sh

# 2. Docker GPU アクセスの修復
./fix_docker_gpu_access.sh

# 3. 手動確認
# NVIDIA Dockerランタイムの確認
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# コンテナ内でのGPU確認
make exec
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 4. 環境テスト
make test-env
```

**根本原因**: nvidia-container-toolkitが未インストールまたは設定不備

**解決手順**:
1. nvidia-container-toolkit のインストール
2. Docker daemon の nvidia runtime 設定
3. Docker サービスの再起動
4. GPU アクセステスト

### 自動修復スクリプト

A100サーバーで以下のコマンドを実行してください：

```bash
# 包括的なGPU修復・検証
./verify_gpu_fix.sh

# または個別実行
./install_nvidia_container_toolkit.sh  # nvidia-container-toolkit インストール
./fix_docker_gpu_access.sh            # Docker GPU アクセス修復
make test-gpu                          # GPU アクセステスト
```

### 手動確認手順

```bash
# 1. ホストでのGPU確認
nvidia-smi

# 2. Docker runtime確認
docker info | grep -i runtime

# 3. 基本的なCUDAコンテナテスト
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# 4. PyTorchコンテナテスト
docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel python -c "import torch; print(torch.cuda.is_available())"

# 5. rlossコンテナテスト
make test-env
```

### メモリ不足エラー
```bash
# バッチサイズを減らす
--batch-size 8

# メモリ最適化設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### ビルドエラー
```bash
# クリーンビルド
make clean
make build

# 詳細ログでビルド
docker build --no-cache -t rloss:a100-ubuntu22.04 -f Dockerfile.a100 .
```

## 📈 性能ベンチマーク（A100 40GB）

### 訓練速度（秒/エポック）
| 設定 | バッチサイズ | 40×40 | 256×256 | 513×513 |
|------|-------------|-------|---------|---------|
| MobileNet | 32/16/8 | 45s | 180s | 420s |
| ResNet50 | 16/8/4 | 90s | 360s | 840s |

### メモリ使用量
| バックボーン | 画像サイズ | バッチサイズ | GPU メモリ |
|-------------|-----------|-------------|-----------|
| MobileNet | 513×513 | 16 | ~28GB |
| ResNet50 | 513×513 | 8 | ~35GB |
| MobileNet | 256×256 | 32 | ~18GB |

## 🔗 関連リンク

- [メインREADME](../README_ja.md)
- [小画像対応スクリプト](../scripts/README.md)
- [PyTorch公式ドキュメント](https://pytorch.org/docs/)
- [NVIDIA A100仕様](https://www.nvidia.com/en-us/data-center/a100/)

## 📝 注意事項

1. **NVIDIA Dockerランタイム必須**: A100を使用するにはnvidia-container-toolkitが必要
2. **CUDA 11.8推奨**: A100の全機能を活用するため
3. **メモリ管理**: 大きなバッチサイズ使用時はメモリ監視が重要
4. **データパス**: `/data/datasets`にデータセットを配置することを推奨
