# Pytorch version of regularized segmentation loss

[日本語版READMEはこちら](../README_ja.md)

## Python environment and dependencies

### Modern PyTorch Setup (推奨)
```bash
# Python 3.8以上を使用
pip install -r requirements.txt
```

### Docker Setup (GPU A6000対応)
```bash
docker build -t rloss:latest .
docker run --gpus all --ipc=host -it --rm -v $(pwd):/workspace rloss:latest
```

### Legacy Setup (pyenv)
```bash
pyenv install 3.8.0
pyenv virtualenv 3.8.0 rloss-env
cd rloss/pytorch
pyenv local rloss-env
pip install -r requirements.txt
```
Other dependencies include [COCOAPI](https://github.com/cocodataset/cocoapi).

## Build python extension module (bilateral filtering拡張モジュールのビルド)

The implementation of DenseCRF loss depends on fast bilateral filtering, which is provided in C++. Use SWIG to wrap C++ for python and then build the python module of bilateral filtering.

```bash
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py build_ext --inplace
python setup.py install
```

### Troubleshooting bilateral filtering build
If you encounter build errors:
```bash
# Clean previous builds
python setup.py clean --all

# Ensure SWIG and OpenMP are installed
sudo apt-get install swig libomp-dev

# Rebuild
swig -python -c++ bilateralfilter.i
python setup.py build_ext --inplace
python setup.py install
```
## denseCRF loss in pytorch

The source code for the denseCRF loss layer is DenseCRFLoss.py. Declare such a loss layer as follows:
```
losslayer=DenseCRFLoss(weight=weight, sigma_rgb=sigma_rgb, sigma_xy=sigma_xy, scale_factor=scale_factor)
```
Here we specify loss weight, Gaussian kernel bandwidth (for RGB and XY), and an optional scale_factor (used to downscale output segmentation so that forward and backward for DenseCRF loss is faster).

The input to the denseCRF loss layer includes image (in the range of [0-255]), segmentation (output of softmax) and a binary tensor specifying region of interest for the regularized loss (e.g. not interested for padded region).
```
losslayer(image,segmentation,region_of_interest)
```
## how to run the code
To train with densecrf loss, use the following example script. The weight of densecrf loss is 2e-9. The bandwidths of Gaussian kernels are 15 and 100 for RGB and XY respectively. Optionally, the output segmentation is downscaled by 0.5 (rloss-scale).
```
python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 
--batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 2 
--densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100
```
(set the path of dataset in [mypath.py](pytorch-deeplab_v3_plus/mypath.py). The path for pascal should have three subdirectories called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble" containing RGB images, groundtruth, and scribbles respectively. PASCAL 2012 segmentation dataset and its scribble annotation can be downloaded via [fetchVOC2012.sh](../data/VOC2012/fetchVOC2012.sh) and [fetchPascalScribble.sh](../data/pascal_scribble/fetchPascalScribble.sh))

## results
<table align="left|center|center|center">
  <tr>
    <td rowspan="2" align="center">network backbone</td>
    <td colspan="2" align="center">weak supervision (~3% pixels labeled)</td>
    <td rowspan="2">full supervision</td>
  </tr>
  <tr>
    <td>(partial) Cross Entropy Loss</td>
    <td>w/ DenseCRF Loss</td>
  </tr>
   <tr>
    <td>mobilenet</td>
    <td>65.8% (1.05 s/it)</td>
     <td><b>69.4%</b> (1.66 s/it)</td>
     <td>72.1% (1.05 s/it)</td>
  </tr>
</table>

**Table 1**: mIOU on PASCAL VOC2012 val set. We report training time for different losses (seconds/iteration, batch_size 12, GTX 1080Ti, AMD FX-6300 3.5GHz).

The trained pytorch models are released <a href="https://cs.uwaterloo.ca/~m62tang/rloss/pytorch" alt=#>here</a>. Example script for testing on one image:
```
python inference.py --backbone mobilenet --checkpoint CHECKPOINT_PATH --image_path IMAGE_PATH --output_directory OUTPUT_DIR
```


## acknowledgement

The code here is built on <a href="https://github.com/jfzhang95/pytorch-deeplab-xception" atl="#">pytorch-deeplab-xception</a>. We alto utilized the efficient c++ implementation of permutohedral lattice from <a href="https://github.com/torrvision/crfasrnn" alt="#">CRF-as-RNN</a>. <a href="http://fangyuliu.me" alt="#">Fangyu Liu</a> from the University of Waterloo helped tremendously in releasing this pytorch version.
