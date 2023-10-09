/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午7:34
# File Name: binary_cuda_functions
# IDE: CLion
*/
#ifndef BINARY_CUDA_FUNCTIONS_CUH
#define BINARY_CUDA_FUNCTIONS_CUH


__global__ void k_num_nbs(float const *const x, float const *const y, float const *const z,
                          float const *const l1norm, int const *const vtx_mapper, float *dev_radius_,
                          const int num_vtx, int *const num_nbs, int* dev_sem_);

__global__ void k_append_neighbours(float const *const x, float const *const y, float const *const z,
                                    float const *const l1norm, int const *const vtx_mapper, int const *const start_pos,
                                    float *dev_radius_, const int num_vtx, int *const neighbours, int *dev_sem_);

__global__ void k_identify_HPs(int const *const num_neighbours,
                                 int *const membership,
                                 const int num_vtx,
                                 int* dev_sem_,
                                 int* dev_min_pts_);
__global__ void k_bfs(bool *const visited, bool *const frontier,
                      int const *const num_nbs,
                      int const *const start_pos,
                      int const *const neighbours,
                      int const *const membership,
                      int num_vtx);

__global__ void cal_mean(float *mean_xd, float *mean_yd, float *mean_zd, float *dev_x_, float *dev_y_, float *dev_z_,
                         int *dev_cluster_idx_, int num_cluster, int num_vtx, int cluster_start);

__global__ void shift_con_clt (int *dev_cluster_idx_, int con_clt_index, int cur_index,int num_vtx_);


__global__ void noise_id_cluster(float *dev_x_, float *dev_y_, float *dev_z_,int *noise_idx_d,int *clt_idx_d,
                                 int *dev_cluster_idx_, int *dev_sem_, int noise_num, int un_noise_num);

__device__ inline float square_dist(const float x1, const float y1, const float z1,
                                    const float x2, const float y2, const float z2);
#endif
