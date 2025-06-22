#!/bin/bash

set -e

echo "=== rloss環境セットアップを開始します ==="

echo "システム依存関係をインストール中..."
sudo apt-get update
sudo apt-get install -y swig build-essential libomp-dev

echo "PyTorch環境をセットアップ中..."
cd pytorch
pip install -r requirements.txt

echo "bilateral filtering拡張モジュールをビルド中..."
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py build_ext --inplace
python setup.py install
cd ../..

echo "データセットディレクトリを作成中..."
mkdir -p /data/datasets/VOCdevkit/VOC2012
mkdir -p /data/datasets/benchmark_RELEASE
mkdir -p /data/datasets/cityscapes
mkdir -p /data/datasets/coco

echo "=== セットアップ完了 ==="
echo "データセットをダウンロードするには:"
echo "  cd data/VOC2012 && ./fetchVOC2012.sh"
echo "  cd data/pascal_scribble && ./fetchPascalScribble.sh"
echo ""
echo "訓練を開始するには:"
echo "  cd pytorch-deeplab_v3_plus"
echo "  python train_withdensecrfloss.py --backbone mobilenet --densecrfloss 2e-9"
