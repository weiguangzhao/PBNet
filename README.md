# PBNet
## Divide and  Conquer: 3D Point Cloud Instance Segmentation With Point-Wise Binarization (ICCV2023)
![overview](https://github.com/weiguangzhao/PBNet/blob/master/doc/overall.png)

## Application: Boxes detection based on our PBNet in simulation scene
[![Boxes detection based on our PBNet in simulation scene](https://res.cloudinary.com/marcomontalbano/image/upload/v1639841022/video_to_markdown/images/youtube--yp7FUmaoW_Q-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=yp7FUmaoW_Q "Boxes detection based on our PBNet in simulation scene")

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
    
## Dataset Preparation

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhao2022divide,
  title={Divide and conquer: 3d point cloud instance segmentation with point-wise binarization},
  author={Zhao, Weiguang and Yan, Yuyao and Yang, Chaolong and Ye, Jianan and Yang, Xi and Huang, Kaizhu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision (ICCV)},
  year={2023}
}
```