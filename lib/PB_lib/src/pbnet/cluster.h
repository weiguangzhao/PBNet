/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午7:34
# File Name: cluster
# IDE: CLion
*/
#ifndef PBNET_CLUSTER_H
#define PBNET_CLUSTER_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

void binary_cluster(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor l1_norm_tensor,
                    at::Tensor index_mapper_tensor, at::Tensor xo_tensor, at::Tensor yo_tensor, at::Tensor zo_tensor,
                    at::Tensor sem_tensor, at::Tensor batch_index_tensor, at::Tensor radius_tensor,
                    at::Tensor min_pts_tensor, at::Tensor cluster_index_tensor, at::Tensor cluster_num_tensor,
                    at::Tensor den_queue_tensor, at::Tensor center_tensor, at::Tensor clt_sem_tensor, int batch_size,
                    float para_f, bool nv_flag);

#endif //PBNET_CLUSTER_H
