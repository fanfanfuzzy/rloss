# Regularized Losses (rloss) による弱教師あり CNN セグメンテーション

(Caffe および PyTorch 実装)

<span align="center"><img src="teaser.png" alt="" width="800"/></span>

弱教師あり学習（例：スクリブル）を使用してCNNによる意味セグメンテーションを訓練するために、正則化損失フレームワークを提案します。
損失は2つの部分から構成されます：スクリブル上の部分交差エントロピー（pCE）損失と、DenseCRFなどの正則化損失です。

このコードを使用する場合は、以下の論文を引用してください。

**"On Regularized Losses for Weakly-supervised CNN Segmentation"** [PDF](http://cs.uwaterloo.ca/~m62tang/OnRegularizedLosses_ECCV18.pdf)</br>
[Meng Tang](http://cs.uwaterloo.ca/~m62tang), [Federico Perazzi](https://fperazzi.github.io/), [Abdelaziz Djelouah](https://adjelouah.github.io/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/), [Christopher Schroers](https://www.disneyresearch.com/people/christopher-schroers/), [Yuri Boykov](https://cs.uwaterloo.ca/about/people/yboykov)</br>
European Conference on Computer Vision (ECCV), Munich, Germany, September 2018.

## 環境構築

### 必要な環境
- Ubuntu 20.04以上
- Python 3.8以上
- CUDA対応GPU（推奨：A6000以上）
- Docker（推奨）

### PyTorch環境のセットアップ

#### 方法1: Dockerを使用（推奨）
```bash
cd pytorch
docker build -t rloss:latest .
docker run --gpus all --ipc=host -it --rm -v $(pwd):/workspace rloss:latest
```

#### 方法2: 直接インストール
```bash
cd pytorch
pip install -r requirements.txt

# bilateral filtering拡張モジュールのビルド
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py build_ext --inplace
python setup.py install
cd ../..
```

#### 方法3: 自動セットアップスクリプト
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### データセットの準備

#### PASCAL VOC2012 + スクリブルアノテーションのダウンロード
```bash
cd data/VOC2012
./fetchVOC2012.sh

cd ../pascal_scribble
./fetchPascalScribble.sh
```

#### データセット構造
```
/data/datasets/VOCdevkit/VOC2012/
├── JPEGImages/          # RGB画像
├── SegmentationClassAug/ # グラウンドトゥルース
└── pascal_2012_scribble/ # スクリブルアノテーション
```

## DenseCRF損失について

DenseCRF損失をCNNに組み込むために、以下の損失レイヤーを追加します。2つの入力を受け取ります：RGB画像と、ソフトセグメンテーション分布です。XY（bi_xy_std）とRGB（bi_rgb_std）のガウシアンカーネルの帯域幅を指定する必要があります。

PyTorchでの実装では、以下のように損失レイヤーを宣言します：
```python
from DenseCRFLoss import DenseCRFLoss

losslayer = DenseCRFLoss(
    weight=weight, 
    sigma_rgb=sigma_rgb, 
    sigma_xy=sigma_xy, 
    scale_factor=scale_factor
)
```

パラメータ：
- `weight`: 損失の重み
- `sigma_rgb`: RGBガウシアンカーネルの帯域幅
- `sigma_xy`: XYガウシアンカーネルの帯域幅  
- `scale_factor`: 出力セグメンテーションのダウンスケール係数（高速化のため）

DenseCRF損失レイヤーへの入力：
```python
losslayer(image, segmentation, region_of_interest)
```
- `image`: [0-255]の範囲のRGB画像
- `segmentation`: softmaxの出力
- `region_of_interest`: 正則化損失の対象領域を指定するバイナリテンソル

## 訓練方法

### 基本的な訓練コマンド
```bash
cd pytorch-deeplab_v3_plus

# DenseCRF損失を使用した訓練
python train_withdensecrfloss.py \
    --backbone mobilenet \
    --lr 0.007 \
    --workers 6 \
    --epochs 60 \
    --batch-size 12 \
    --checkname deeplab-mobilenet \
    --eval-interval 2 \
    --dataset pascal \
    --save-interval 2 \
    --densecrfloss 2e-9 \
    --rloss-scale 0.5 \
    --sigma-rgb 15 \
    --sigma-xy 100
```

### 2段階訓練プロセス

1. **第1段階**: 部分交差エントロピー損失のみで訓練
   - PASCAL VOC12 valセットでmIOU ~55.8%を達成

2. **第2段階**: DenseCRF損失を追加してファインチューニング  
   - mIOUが~62.3%に向上

### 推論
```bash
python inference.py \
    --backbone mobilenet \
    --checkpoint CHECKPOINT_PATH \
    --image_path IMAGE_PATH \
    --output_directory OUTPUT_DIR
```

## 性能結果

<table align="center">
  <tr>
    <td rowspan="2" align="center">ネットワークバックボーン</td>
    <td colspan="2" align="center">弱教師あり学習（~3%のピクセルがラベル付き）</td>
    <td rowspan="2">完全教師あり学習</td>
  </tr>
  <tr>
    <td>（部分）交差エントロピー損失</td>
    <td>DenseCRF損失付き</td>
  </tr>
   <tr>
    <td>mobilenet</td>
    <td>65.8% (1.05 s/it)</td>
     <td><b>69.4%</b> (1.66 s/it)</td>
     <td>72.1% (1.05 s/it)</td>
  </tr>
</table>

**表1**: PASCAL VOC2012 valセットでのmIOU。異なる損失での訓練時間を報告（秒/イテレーション、batch_size 12、GTX 1080Ti、AMD FX-6300 3.5GHz）。

### PASCAL VOC2012での詳細結果

<table align="center">
  <tr>
    <td rowspan="2" align="center">ネットワーク</td>
    <td colspan="2" align="center">弱教師あり学習（~3%のピクセルがラベル付き）</td>
    <td rowspan="2">完全教師あり学習</td>
  </tr>
  <tr>
    <td>（部分）交差エントロピー損失</td>
    <td>DenseCRF損失付き</td>
  </tr>
   <tr>
    <td>Deeplab_largeFOV</td>
    <td>55.8%</td>
     <td><b>62.3%</b></td>
     <td>63.0%</td>
  </tr>
     <tr>
    <td>Deeplab_Msc_largeFOV</td>
    <td>n/a</td>
     <td><b>63.2%</b></td>
     <td>64.1%</td>
  </tr>
  <tr>
    <td>Deeplab_VGG16</td>
    <td>60.7%</td>
     <td><b>64.7%</b></td>
     <td>68.8%</td>
  </tr>
  <tr>
    <td>Deeplab_ResNet101</td>
    <td>69.5%</td>
     <td><b>73.0%</b></td>
     <td>75.6%</td>
  </tr>
</table>

## トラブルシューティング

### よくある問題

1. **SWIG not found エラー**
   ```bash
   sudo apt-get install swig
   ```

2. **OpenMP エラー**
   ```bash
   sudo apt-get install libomp-dev
   ```

3. **CUDA out of memory**
   - バッチサイズを減らす: `--batch-size 8`
   - スケールファクターを小さくする: `--rloss-scale 0.25`

4. **bilateral filtering import エラー**
   ```bash
   cd wrapper/bilateralfilter
   python setup.py clean --all
   swig -python -c++ bilateralfilter.i
   python setup.py build_ext --inplace
   python setup.py install
   ```

5. **numpy互換性エラー**
   - 新しいnumpyバージョンでは`get_numpy_include()`が廃止されています
   - setup.pyが自動的に適切なメソッドを選択します

6. **GPU利用不可エラー**
   ```bash
   # CUDA利用可能性の確認
   python -c "import torch; print(torch.cuda.is_available())"
   
   # GPU情報の確認
   nvidia-smi
   ```

### パフォーマンス最適化

1. **メモリ使用量の削減**
   - `--rloss-scale`を小さくする（0.25-0.5）
   - バッチサイズを調整する
   - 勾配累積を使用する

2. **訓練速度の向上**
   - 複数GPUを使用する: `--gpu-ids 0,1,2,3`
   - 同期バッチ正規化を有効にする: `--sync-bn True`
   - ワーカー数を調整する: `--workers 8`

## その他の正則化損失

このフレームワークでは、セグメンテーション用の任意の微分可能な正則化損失を使用できます（例：正規化カット、サイズ制約など）。

関連論文：
- **"Normalized Cut Loss for Weakly-supervised CNN Segmentation"** [PDF](https://cs.uwaterloo.ca/~m62tang/ncloss_CVPR18.pdf)
- **"Size-constraint loss for weakly supervised CNN segmentation"** [PDF](https://arxiv.org/pdf/1805.04628.pdf) [Code](https://github.com/LIVIAETS/SizeLoss_WSS)

## 事前訓練済みモデル

訓練済みのPyTorchモデルは[こちら](https://cs.uwaterloo.ca/~m62tang/rloss/pytorch)で公開されています。

## 謝辞

このコードは[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)をベースに構築されています。また、[CRF-as-RNN](https://github.com/torrvision/crfasrnn)の効率的なpermutohedral latticeのC++実装を利用しています。ウォータールー大学の[Fangyu Liu](http://fangyuliu.me)氏がこのPyTorchバージョンのリリースに多大な貢献をしました。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 更新履歴

- **2025年6月**: PyTorch 2.x対応、GPU A6000サポート、日本語ドキュメント追加
- **2019年**: 初期PyTorch実装リリース
- **2018年**: 論文発表とCaffe実装リリース
