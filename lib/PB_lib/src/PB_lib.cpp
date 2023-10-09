/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/10/31  下午7:12
# File Name: PB_lib
# IDE: CLion
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "iou/get_iou.cpp"
#include "cal_iou_and_masklabel/cal_iou_and_masklabel.cpp"