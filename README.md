# PBNet
## Divide and  Conquer: 3D Point Cloud Instance Segmentation With Point-Wise Binarization (ICCV2023)
![overview](https://github.com/weiguangzhao/PBNet/blob/master/doc/overall.png)

## Application: [Detect Adjacent Boxes](https://www.youtube.com/watch?v=yp7FUmaoW_Q)

## Environments
This code could be run on RTX8000 RTX3090 RTX2080TI etc. with CUDA11.x and CUDA 10.X. Below we take RTX3090 environments 
as an example. You need at least two RTX3090 cards with 24GB.
### Creat Conda Environment
    conda create -n pbnet python=3.8
    conda activate pbnet

### Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
    conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision
    conda install openblas-devel -c anaconda
    
    # Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
    # export CUDA_HOME=/usr/local/cuda-11.1
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
    
    # Or if you want local MinkowskiEngine
    cd lib
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

### Install Our PB_lib
    pip install -r requirements
    cd lib/PB_lib
    python setup.py devel

### Install segmentator 
```
cd lib/segmentator
cd csrc && mkdir build && cd build
conda install cmake cudnn

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`

make && make install # after install, please do not delete this folder (as we only create a symbolic link)
```

Further segmentator information can be found in [DKNet](https://github.com/W1zheng/DKNet) and [Segmentator](https://github.com/Karbo123/segmentator).
    
## Dataset Preparation
(1) Download the [ScanNet v2](http://www.scan-net.org/) dataset.

(2) Put the data in the corresponding folders. The dataset files are organized as follows.
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`,  `[scene_id].aggregation.json`  and `[scene_id].txt`  into the `datasets/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).

* Copy the files `[scene_id]_vh_clean_2.ply` into the `datasets/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).

* Put the file `scannetv2-labels.combined.tsv` in the `datasets/scannetv2` folder.


```
PBNet
├── datasets
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json & [scene_id].txt
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json & [scene_id].txt
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```
(3) Decode the files to the "PBNet/datasets/scannetv2/npy/"
    
    cd PBNet
    export PYTHONPATH=./
    python datasets/scannetv2/decode_scannet.py

 
## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhao2023divide,
  title={Divide and conquer: 3d point cloud instance segmentation with point-wise binarization},
  author={Zhao, Weiguang and Yan, Yuyao and Yang, Chaolong and Ye, Jianan and Yang, Xi and Huang, Kaizhu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision (ICCV)},
  year={2023}
}
```