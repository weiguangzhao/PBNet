/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午9:36
# File Name: binary_cuda_functions
# IDE: CLion
*/

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <math.h>
#include "binary_cuda_functions.cuh"

constexpr int kSharedMemBytes = 48 * 1024;

/*!
 * Calculate the number of neighbours of each vertex. One kernel thread per
 * vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param z - z values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param rad - radius.
 * @param num_vtx - number of vertices.
 * @param num_nbs - output array.
 */
__global__ void k_num_nbs(float const *const x, float const *const y, float const *const z,
                          float const *const l1norm, int const *const vtx_mapper, float *dev_radius_,
                          const int num_vtx, int *const num_nbs, int *dev_sem_) {
    int const thread_index = blockIdx.x * blockDim.x + threadIdx.x ;
    if (thread_index >= num_vtx) return;

    int cur_sem = dev_sem_[thread_index] -2;
    float cur_radius = dev_radius_[cur_sem];


    // first vtx of current block.
    const int tb_start = blockIdx.x * blockDim.x;
    // last vtx of current block.
    const int tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

    int land_id = threadIdx.x & 0x1f;
    float const *possible_range_start, *possible_range_end;
    if (land_id == 0) {
        // inclusive start
        // https://github.com/NVIDIA/thrust/issues/1734
        possible_range_start = thrust::lower_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * cur_radius);
        // exclusive end
        possible_range_end = thrust::upper_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * cur_radius);
    }
    possible_range_start =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
    possible_range_end =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

    // the number of threads might not be blockDim.x, if this is the last block.
    int const num_threads = tb_end - tb_start + 1;
    const int tile_size = kSharedMemBytes / 4 / (1 + 1 + 1);
    // first half of shared stores Xs; second half stores Ys; third half stores Ys.
    __shared__ float shared[tile_size * (1 + 1 + 1)];
    auto *const sh_x = shared;
    auto *const sh_y = shared + tile_size;
    auto *const sh_z = shared + tile_size*2;
    int ans = 0;

    for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
         curr_ptr += tile_size) {
        // curr_ptr's index
        int const curr_idx = curr_ptr - l1norm;
        // current range; might be less than tile_size.
        int const curr_range = min(tile_size, static_cast<int>(possible_range_end - curr_ptr));
        // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
        // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
        // ...
        // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
        __syncthreads();
        for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
            sh_x[i] = x[curr_idx + i];
            sh_y[i] = y[curr_idx + i];
            sh_z[i] = z[curr_idx + i];
        }
        __syncthreads();
        const float thread_x = x[thread_index], thread_y = y[thread_index], thread_z = z[thread_index];
        for (auto j = 0; j < curr_range; ++j) {
            ans += square_dist(thread_x, thread_y, thread_z, sh_x[j], sh_y[j], sh_z[j]) <=cur_radius * cur_radius;
        }
    }
    num_nbs[vtx_mapper[thread_index]] = ans - 1;
}
/*!
 * Populate the neighbours array. One kernel thread per vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param start_pos - neighbours starting index of each vertex.
 * @param rad - radius.
 * @param num_vtx - number of vertices
 * @param neighbours - output array
 */
__global__ void k_append_neighbours(float const *const x, float const *const y, float const *const z,
                                    float const *const l1norm,
                                    int const *const vtx_mapper,
                                    int const *const start_pos,
                                    float *dev_radius_, const int num_vtx,
                                    int *const neighbours, int *dev_sem_) {
    int const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_vtx) return;

    int cur_sem = dev_sem_[thread_index] -2;
    float cur_radius = dev_radius_[cur_sem];


    // first vtx of current block.
    const int tb_start = blockIdx.x * blockDim.x;
    // last vtx of current block.
    const int tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

    int land_id = threadIdx.x & 0x1f;
    float const *possible_range_start, *possible_range_end;
    if (land_id == 0) {
        // inclusive start
        possible_range_start = thrust::lower_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * cur_radius);
        // exclusive end
        possible_range_end = thrust::upper_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * cur_radius);
    }
    possible_range_start =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
    possible_range_end =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

    int const num_threads = tb_end - tb_start + 1;
    // different from previous kernel, here the shared array is tri-partitioned,
    // because of the frequent access to vtx_mapper.
    const int tile_size = kSharedMemBytes / 4 / (1 + 1 + 1 + 1);
    __shared__ float shared[tile_size * (1 + 1 + 1 + 1)];
    auto *const sh_x = shared;
    auto *const sh_y = shared + tile_size;
    auto *const sh_z = shared + tile_size*2;
    auto *const sh_vtx_mapper = (int *)(sh_z + tile_size);
    int upos = start_pos[vtx_mapper[thread_index]];

    for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end; curr_ptr += tile_size) {
        // curr_ptr's index
        int const curr_idx = curr_ptr - l1norm;
        // current range; might be less than tile_size.
        int const curr_range =min(tile_size, static_cast<int>(possible_range_end - curr_ptr));
        // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
        // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
        // ...
        // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
        __syncthreads();
        for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
            sh_x[i] = x[curr_idx + i];
            sh_y[i] = y[curr_idx + i];
            sh_z[i] = z[curr_idx + i];
            sh_vtx_mapper[i] = vtx_mapper[curr_idx + i];
        }
        __syncthreads();
        const float thread_x = x[thread_index], thread_y = y[thread_index], thread_z = z[thread_index];
        for (auto j = 0; j < curr_range; ++j) {
            if (thread_index != curr_idx + j && square_dist(thread_x, thread_y, thread_z,
                                                            sh_x[j], sh_y[j], sh_z[j]) <= cur_radius * cur_radius) {
                neighbours[upos++] = sh_vtx_mapper[j];
            }
        }
    }
}
/*!
 * Identify all the Core vertices. One kernel thread per vertex.
 * @param num_neighbours - the number of neighbours of each vertex.
 * @param membership - membership of each vertex.
 * @param num_vtx - number of vertex.
 * @param min_pts - query parameter, minimum number of points to be consider as
 * a Core.
 */
__global__ void k_identify_HPs(int const *const num_neighbours,
                                 int *const membership,
                                 const int num_vtx,
                                 int* dev_sem_,
                                 int* dev_min_pts_){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx) return;
    membership[u] = 2;
    int cur_sem = dev_sem_[u] -2;
    int cur_min_pts = dev_min_pts_[cur_sem];
    if (num_neighbours[u] >= cur_min_pts) membership[u] = 0;
}
/*!
 * Traverse the graph from each vertex. One kernel thread per vertex.
 * @param visited - boolean array that tracks if a vertex has been visited.
 * @param frontier - boolean array that tracks if a vertex is on the frontier.
 * @param num_nbs - the number of neighbours of each vertex.
 * @param start_pos - neighbours starting index of each vertex.
 * @param neighbours - the actually neighbour indices of each vertex.
 * @param membership - membership of each vertex.
 * @param num_vtx - number of vertices of the graph.
 */
__global__ void k_bfs(bool *const visited, bool *const frontier,
                      int const *const num_nbs,
                      int const *const start_pos,
                      int const *const neighbours,
                      int const *const membership,
                      int num_vtx) {
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx) return;
    if (!frontier[u]) return;
    frontier[u] = false;
    visited[u] = true;
    // Stop BFS if u is not Core.
    if (membership[u] != 0) return;
    int u_start = start_pos[u];
    for (int i = 0; i < num_nbs[u]; ++i) {
        int v = neighbours[u_start + i];
        if (!visited[v]) frontier[v] = true;
    }
}

__global__ void cal_mean(float *mean_xd, float *mean_yd, float *mean_zd, float *dev_x_, float *dev_y_, float *dev_z_,
                         int *dev_cluster_idx_, int num_cluster, int num_vtx, int cluster_start){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_cluster) return;
    int cluster_cur=0;
    cluster_cur = cluster_start + u;
    int N=0;
    float M_x=0;
    float M_y=0;
    float M_z=0;
    float x=0;
    float y=0;
    float z=0;
    for(int i =0; i<num_vtx; i++){
        if(dev_cluster_idx_[i] == cluster_cur){
            N++;
            x = dev_x_[i];
            y = dev_y_[i];
            z = dev_z_[i];

            M_x= M_x+(x- M_x)/N;
            M_y= M_y+(y- M_y)/N;
            M_z= M_z+(z- M_z)/N;
        }
    }
//    if(N<2) printf("N is less than 2\n");
    mean_xd[u] =M_x;
    mean_yd[u] =M_y;
    mean_zd[u] =M_z;
}


__global__ void shift_con_clt (int *dev_cluster_idx_, int con_clt_index, int cur_index, int num_vtx_){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx_) return;
    if (dev_cluster_idx_[u] == con_clt_index)
        dev_cluster_idx_[u] = cur_index;
    if (dev_cluster_idx_[u] > con_clt_index)
        dev_cluster_idx_[u]--;
}

__global__ void noise_id_cluster(float *dev_x_, float *dev_y_, float *dev_z_,int *noise_idx_d,int *clt_idx_d,
                                     int *dev_cluster_idx_, int *dev_sem_, int noise_num, int un_noise_num){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= noise_num) return;
    int noise_real_idx = noise_idx_d[u];
    float n_x,n_y,n_z;
    n_x= dev_x_[noise_real_idx];
    n_y= dev_y_[noise_real_idx];
    n_z= dev_z_[noise_real_idx];
    int sem_noise = dev_sem_[noise_real_idx];

    float un_x, un_y,un_z, dist, min_dist;
    int min_index =0;
    int clt_real_idx=0;
    int count_i =0;
    for(int i=0;i<un_noise_num;i++){
        clt_real_idx=clt_idx_d[i];
        if (dev_sem_[clt_real_idx]!=sem_noise) continue;
        un_x= dev_x_[clt_real_idx];
        un_y= dev_y_[clt_real_idx];
        un_z= dev_z_[clt_real_idx];
        dist = square_dist(n_x,n_y,n_z,un_x,un_y,un_z);
        if(count_i==0) min_dist=dist;
        count_i ++;
        if (dist<=min_dist){
            min_dist = dist;
            min_index = clt_real_idx;
        }
    }
    if (count_i==0) {
        for(int i=0;i<un_noise_num;i++){
            un_x= dev_x_[clt_real_idx];
            un_y= dev_y_[clt_real_idx];
            un_z= dev_z_[clt_real_idx];
            dist = square_dist(n_x,n_y,n_z,un_x,un_y,un_z);
            if(count_i==0) min_dist=dist;
            count_i ++;
            if (dist<=min_dist){
                min_dist = dist;
                min_index = clt_real_idx;
            }
        }
    }
    dev_cluster_idx_[noise_real_idx] = dev_cluster_idx_[min_index];
}


__device__ inline float square_dist(const float x1, const float y1, const float z1,
                                    const float x2, const float y2, const float z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

