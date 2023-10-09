/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/12/8
# File Name: cal_normal
# IDE: CLion
*/

#include <fstream>  // for file stream
#include "cal_normal.h"
#include "../pbnet/binary.cuh"

using namespace std;
//================np.linalg.norm()  3dim=============================
__device__ inline float linalg_two_norm(const float x1, const float y1, const float z1) {
    return sqrt(x1*x1 + y1*y1 + z1*z1);
}


//================np.dot()  3dim=============================
__device__ inline float dot_three_d(const float x1, const float y1, const float z1,
                                    const float x2, const float y2, const float z2) {
    return (x1*x2 + y1*y2+ z1*z2);
}

//================np.cross()  3dim=============================
__device__ inline float cross_x(const float x1, const float y1, const float z1,
                                const float x2, const float y2, const float z2) {
    return (y1*z2 - z1*y2);
}

__device__ inline float cross_y(const float x1, const float y1, const float z1,
                                const float x2, const float y2, const float z2) {
    return (z1*x2 - x1*z2);
}

__device__ inline float cross_z(const float x1, const float y1, const float z1,
                                const float x2, const float y2, const float z2) {
    return (x1*y2 - y1*x2);
}

__global__ void surface_normal_area(float *dev_xyz,  int *dev_face, int num_face, float *dev_normal, float *dev_area) {
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_face) return;
    int index_a = dev_face[u*3 + 0];
    int index_b = dev_face[u*3 + 1];
    int index_c = dev_face[u*3 + 2];

    float a_x=0, a_y=0, a_z=0;
    float b_x=0, b_y=0, b_z=0;
    a_x = dev_xyz[index_b*3 +0] - dev_xyz[index_a*3 +0];
    a_y = dev_xyz[index_b*3 +1] - dev_xyz[index_a*3 +1];
    a_z = dev_xyz[index_b*3 +2] - dev_xyz[index_a*3 +2];

    b_x = dev_xyz[index_c*3 +0] - dev_xyz[index_a*3 +0];
    b_y = dev_xyz[index_c*3 +1] - dev_xyz[index_a*3 +1];
    b_z = dev_xyz[index_c*3 +2] - dev_xyz[index_a*3 +2];

    dev_normal[u*3 + 0] = cross_x(a_x, a_y, a_z, b_x, b_y, b_z);
    dev_normal[u*3 + 1] = cross_y(a_x, a_y, a_z, b_x, b_y, b_z);
    dev_normal[u*3 + 2] = cross_z(a_x, a_y, a_z, b_x, b_y, b_z);

    dev_area[u] = dot_three_d(dev_normal[u*3 + 0], dev_normal[u*3 + 1], dev_normal[u*3 + 2],
                              dev_normal[u*3 + 0], dev_normal[u*3 + 1], dev_normal[u*3 + 2]);
    dev_area[u] = dev_area[u] / 2.0;

    float norm_l2 = 0;
    norm_l2 = linalg_two_norm(dev_normal[u*3 + 0], dev_normal[u*3 + 1], dev_normal[u*3 + 2]);

    dev_normal[u*3 + 0] = dev_normal[u*3 + 0]/norm_l2;
    dev_normal[u*3 + 1] = dev_normal[u*3 + 1]/norm_l2;
    dev_normal[u*3 + 2] = dev_normal[u*3 + 2]/norm_l2;

}

__global__ void vertex_normal(int *dev_face, int num_vtx, float *dev_normal, float *dev_area, float *dev_normal_xyz,
                              int num_face) {
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx) return;
    float sum_adj_normal_x = 0;
    float sum_adj_normal_y = 0;
    float sum_adj_normal_z = 0;
    float sum_adj_area = 0;
    float norm_l2 = 0;

    for(int i =0; i<num_face; i++){
        if (dev_face[i*3 + 0]==u || dev_face[i*3 + 1]==u || dev_face[i*3 + 2]==u){
            sum_adj_normal_x = sum_adj_normal_x + dev_normal[i*3 +0] * dev_area[i];
            sum_adj_normal_y = sum_adj_normal_y + dev_normal[i*3 +1] * dev_area[i];
            sum_adj_normal_z = sum_adj_normal_z + dev_normal[i*3 +2] * dev_area[i];
            sum_adj_area = sum_adj_area + dev_area[i];
        }
    }

    if(sum_adj_area==0.0) {
        dev_normal_xyz[u*3 +0] = 0.0;
        dev_normal_xyz[u*3 +1] = 0.0;
        dev_normal_xyz[u*3 +2] = 1.0;
    } else{
        sum_adj_normal_x = sum_adj_normal_x/sum_adj_area;
        sum_adj_normal_y = sum_adj_normal_y/sum_adj_area;
        sum_adj_normal_z = sum_adj_normal_z/sum_adj_area;
        norm_l2 = linalg_two_norm(sum_adj_normal_x, sum_adj_normal_y, sum_adj_normal_z);
        dev_normal_xyz[u*3 +0] = sum_adj_normal_x/norm_l2;
        dev_normal_xyz[u*3 +1] = sum_adj_normal_y/norm_l2;
        dev_normal_xyz[u*3 +2] = sum_adj_normal_z/norm_l2;
    }
}

void cal_normal_line(at::Tensor xyz_tensor, at::Tensor f_abc_tensor, at::Tensor n_xyz_tensor, int num_vtx, int num_face) {

    // =================get C++ pointer================
    float *xyz = xyz_tensor.data<float>();
    int *face_abc = f_abc_tensor.data<int>();
    float *normal_xyz = n_xyz_tensor.data<float>();

    const auto N_s = sizeof(xyz[0]) * num_vtx;
    const auto M_s = sizeof(xyz[0]) * num_face;
    const auto L_s = sizeof(int) * num_face;

    // Malloc device pointer
    float *dev_xyz, *dev_normal, *dev_area, *dev_normal_xyz;
    int *dev_face;
    CUDA_ERR_CHK( cudaMalloc((void **) &dev_xyz, N_s*3));
    CUDA_ERR_CHK( cudaMalloc((void **) &dev_normal, M_s*3));
    CUDA_ERR_CHK( cudaMalloc((void **) &dev_area, M_s));
    CUDA_ERR_CHK( cudaMalloc((void **) &dev_normal_xyz, N_s*3));
    CUDA_ERR_CHK( cudaMalloc((void **) &dev_face, L_s*3));

    CUDA_ERR_CHK(cudaMemcpy(dev_xyz, xyz, N_s*3, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(dev_face, face_abc, L_s*3, cudaMemcpyHostToDevice));

    dim3 block_face(DIVUP(num_face, THREADS_PER_BLOCK));
    dim3 thread_face(THREADS_PER_BLOCK);

    surface_normal_area<<<block_face, thread_face>>>(dev_xyz, dev_face, num_face, dev_normal, dev_area);
    CUDA_ERR_CHK(cudaDeviceSynchronize());
    CUDA_ERR_CHK(cudaPeekAtLastError());

    dim3 block_vtx(DIVUP(num_vtx, THREADS_PER_BLOCK));
    dim3 thread_vtx(THREADS_PER_BLOCK);

    vertex_normal<<<block_vtx, thread_vtx>>>(dev_face, num_vtx, dev_normal, dev_area, dev_normal_xyz, num_face);
    CUDA_ERR_CHK(cudaDeviceSynchronize());
    CUDA_ERR_CHK(cudaPeekAtLastError());

    CUDA_ERR_CHK(cudaMemcpy(normal_xyz, dev_normal_xyz, N_s*3, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHK(cudaFree(dev_xyz));
    CUDA_ERR_CHK(cudaFree(dev_normal));
    CUDA_ERR_CHK(cudaFree(dev_area));
    CUDA_ERR_CHK(cudaFree(dev_normal_xyz));
    CUDA_ERR_CHK(cudaFree(dev_face));

}






