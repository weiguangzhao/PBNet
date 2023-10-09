/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午7:34
# File Name: binary
# IDE: CLion
*/
#include <thrust/count.h>
#include <fstream>
#include <vector>
#include <algorithm>

#include "binary.cuh"
#include "binary_cuda_functions.cuh"

using namespace std;

//================================================= define input style==============================
BINARY::Solver::Solver(float *x_tensor, float *y_tensor, float *z_tensor,float *l1_norm_tensor,
                       int *index_mapper_tensor, float *xo_tensor, float *yo_tensor, float *zo_tensor, int *sem_tensor,
                       int num, float *radius_tensor, int *min_pts_tensor, int *cluster_id_tensor,
                       int *den_queue_tensor){
    num_vtx_= num;
    num_blocks_ = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));

    x_ = x_tensor;
    y_ = y_tensor;
    z_ = z_tensor;
    l1norm_ = l1_norm_tensor;
    vtx_mapper_ = index_mapper_tensor;

    xo_ = xo_tensor;
    yo_ = yo_tensor;
    zo_ = zo_tensor;
    sem_ = sem_tensor;
    min_pts_ = min_pts_tensor;
    radius_ = radius_tensor;

    cluster_ids = cluster_id_tensor;
    den_queue_ = den_queue_tensor;

    //=====init membership
    vector<int> memberships_init;
    memberships_init.resize(num_vtx_, -1);
    memberships = (int *)malloc(num_vtx_*sizeof(int));
    memcpy(memberships, &memberships_init[0], memberships_init.size()*sizeof(int));
}

void BINARY::Solver::sort_input_by_l1norm() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_vtx_mapper_[0]) * num_vtx_;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_z_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_l1norm_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_vtx_mapper_, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x_, x_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_y_, y_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_z_, z_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_l1norm_, l1norm_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_vtx_mapper_, vtx_mapper_, K, H2D));

  // https://thrust.github.io/doc/classthrust_1_1zip__iterator.html
  typedef typename thrust::tuple<float *, float *, float *, int *> IteratorTuple;
  typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator begin(thrust::make_tuple(dev_x_, dev_y_, dev_z_, dev_vtx_mapper_));
  thrust::sort_by_key(thrust::device, dev_l1norm_, dev_l1norm_ + num_vtx_, begin);
}

void BINARY::Solver::calc_num_neighbours() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_num_neighbours_[0]) * num_vtx_;
  const auto M = sizeof(dev_membership_[0]) * num_vtx_;
  const auto M_I = sizeof(x_[0]) * 18;

  int last_vtx_num_nbs = 0;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_neighbours_, K));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_radius_, M_I));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_sem_, M));
  CUDA_ERR_CHK(cudaMemcpy(dev_radius_, radius_, M_I, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_sem_, sem_, M, H2D));

  k_num_nbs<<<num_blocks_, BLOCK_SIZE>>>(dev_x_, dev_y_, dev_z_, dev_l1norm_, dev_vtx_mapper_,
                                         dev_radius_, num_vtx_, dev_num_neighbours_, dev_sem_);
  CUDA_ERR_CHK(cudaPeekAtLastError());
  //std::cout << "Address of K: " << dev_num_neighbours_ << std::endl;
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_num_nbs, dev_num_neighbours_ + num_vtx_ - 1, sizeof(last_vtx_num_nbs), D2H));
  total_num_nbs_ += last_vtx_num_nbs;
}

void BINARY::Solver::calc_start_pos() {
  int last_vtx_start_pos = 0;

  const auto N = sizeof(dev_start_pos_[0]) * num_vtx_;
  const auto K = sizeof(dev_num_neighbours_[0]) * num_vtx_;

  // Do not free dev_start_pos_. It's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos_, N));
  thrust::exclusive_scan(thrust::device, dev_num_neighbours_, dev_num_neighbours_ + num_vtx_, dev_start_pos_);
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_start_pos, dev_start_pos_ + num_vtx_ - 1, sizeof(int), D2H));
  total_num_nbs_ += last_vtx_start_pos;
}

void BINARY::Solver::append_neighbours() {

  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_start_pos_[0]) * num_vtx_;
  const auto J = sizeof(dev_neighbours_[0]) * total_num_nbs_;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours_, J));

  k_append_neighbours<<<num_blocks_, BLOCK_SIZE>>>(dev_x_, dev_y_, dev_z_, dev_l1norm_, dev_vtx_mapper_,
                                                   dev_start_pos_, dev_radius_, num_vtx_, dev_neighbours_, dev_sem_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  // dev_x_ and dev_y_ are no longer used.
  CUDA_ERR_CHK(cudaFree(dev_x_));
  CUDA_ERR_CHK(cudaFree(dev_y_));
  CUDA_ERR_CHK(cudaFree(dev_z_));
  // graph has been fully constructed, hence free all the sorting related.
  CUDA_ERR_CHK(cudaFree(dev_l1norm_));
  CUDA_ERR_CHK(cudaFree(dev_vtx_mapper_));
  // dev_num_neighbours_, dev_start_pos_, dev_neighbours_ in GPU RAM.
  CUDA_ERR_CHK(cudaFree(dev_radius_));
  //  CUDA_ERR_CHK(cudaFree(dev_sem_));

}

void BINARY::Solver::identify_HPs() {
  const auto N = sizeof(dev_num_neighbours_[0]) * num_vtx_;
  const auto M = sizeof(dev_membership_[0]) * num_vtx_;
  const auto J = sizeof(dev_neighbours_[0]) * total_num_nbs_;
  const auto M_I = sizeof(dev_membership_[0]) * 18;

  // Do not free dev_membership_ as it's used in BFS. The content will be
  // copied out at the end of |identify_clusters|.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_membership_, M));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_min_pts_, M_I));
  //  CUDA_ERR_CHK(cudaMalloc((void **)&dev_sem_, M));
  CUDA_ERR_CHK(cudaMemcpy(dev_min_pts_, min_pts_, M_I, H2D));
  //  CUDA_ERR_CHK(cudaMemcpy(dev_sem_, sem_, M, H2D));

  k_identify_HPs<<<num_blocks_, BLOCK_SIZE>>>(dev_num_neighbours_, dev_membership_, num_vtx_, dev_sem_, dev_min_pts_);
  // Copy the membership data from GPU to CPU RAM as needed for the BFS
  // condition check.
  CUDA_ERR_CHK(cudaMemcpy(memberships, dev_membership_, M, D2H));
  CUDA_ERR_CHK(cudaMemcpy(den_queue_, dev_num_neighbours_, M, D2H));
  CUDA_ERR_CHK(cudaFree(dev_min_pts_));
  CUDA_ERR_CHK(cudaFree(dev_sem_));

}

int BINARY::Solver::identify_clusters(int cluster_accum) {
  const auto T = sizeof(bool) * num_vtx_;
  const auto N = sizeof(dev_num_neighbours_[0]) * num_vtx_;
  const auto L = sizeof(dev_membership_[0]) * num_vtx_;
  const auto K = sizeof(dev_neighbours_[0]) * total_num_nbs_;

  int cluster = cluster_accum;
  for (int u = 0; u < num_vtx_; ++u) {
    if (cluster_ids[u] == -1 && memberships[u] == 0) {
      bfs_sem(u, cluster);
      ++cluster;
    }
  }
  CUDA_ERR_CHK(cudaFree(dev_num_neighbours_));
  CUDA_ERR_CHK(cudaFree(dev_start_pos_));
  CUDA_ERR_CHK(cudaFree(dev_neighbours_));
  CUDA_ERR_CHK(cudaFree(dev_membership_));
  free(memberships);

  return cluster;
}

void BINARY::Solver::bfs_sem(const int u, const int cluster) {
    auto visited = new bool[num_vtx_]();
    auto frontier = new bool[num_vtx_]();
    int num_frontier = 1;
    frontier[u] = true;
    const auto T = sizeof(visited[0]) * num_vtx_;
    const auto L = sizeof(*dev_membership_) * num_vtx_;

    int sem_cur = 0;
    sem_cur = sem_[u];
//    printf("test sem_flg\n");

    bool *dev_visited, *dev_frontier;
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_frontier, T));
    CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_frontier, frontier, T, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_membership_, memberships, L, H2D));

    while (num_frontier > 0) {
        k_bfs<<<num_blocks_, BLOCK_SIZE>>>(dev_visited, dev_frontier, dev_num_neighbours_, dev_start_pos_,
                                           dev_neighbours_, dev_membership_, num_vtx_);
        CUDA_ERR_CHK(cudaPeekAtLastError());
        num_frontier = thrust::count(thrust::device, dev_frontier, dev_frontier + num_vtx_, true);
    }
    // we don't care about he content in dev_frontier now, hence no need to copy back.
    CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, D2H));
    CUDA_ERR_CHK(cudaFree(dev_visited));
    CUDA_ERR_CHK(cudaFree(dev_frontier));

    for (int n = 0; n < num_vtx_; ++n) {
        if (visited[n] && sem_[n] == sem_cur) {
            cluster_ids[n] = cluster;
            if (memberships[n] != 0) {
                memberships[n] = 1;
            }
        }
    }

    delete[] visited;
    delete[] frontier;
}

int BINARY::Solver::filter(int num_cluster, int cluster_start, float para_f, vector<int>& clt_sem_vector){

    const auto N = sizeof(x_[0]) * num_vtx_;
    const auto M = sizeof(x_[0]) * num_cluster;

    CUDA_ERR_CHK(cudaMalloc((void **)&dev_cluster_idx_, N));
    // Memcpy from host to device
    CUDA_ERR_CHK(cudaMemcpy(dev_cluster_idx_, cluster_ids, N, H2D));

    // mean count from HAIS
    float mean_count_[] ={3917.0, 12056.0, 2303.0, 8331.0, 3948.0, 3166.0, 5629.0, 11719.0,1003.0, 3317.0, 4912.0, 10221.0, 3889.0, 4136.0, 2120.0, 945.0, 3967.0, 2589.0};

    // calculate the point number of cluster
    vector<int> clt_sem;
    vector<int> clt_num;

    for (int i=0; i < num_cluster; i++){
        clt_sem.push_back(0);
        clt_num.push_back(0);
    }

    int cluster_id = 0;
    for(int i=0; i<num_vtx_; i++){
        cluster_id = cluster_ids[i];
        if (cluster_id != -1){
            clt_num[cluster_id-cluster_start]++;
            clt_sem[cluster_id-cluster_start] = sem_[i];
        }
    }

    int reduce_count = 0;
    int reduce_clt_idx=0;
    float cur_mean_count =0.0;
    for (int i=0; i<num_cluster; i++){
        int cur_sem = clt_sem[i] - 2;
        clt_sem_vector.push_back(clt_sem[i]);
        reduce_clt_idx = i + cluster_start -reduce_count;
        cur_mean_count = mean_count_[cur_sem] * para_f;
        if (float(clt_num[i]) < cur_mean_count){
            shift_con_clt<<<num_blocks_, BLOCK_SIZE>>>(dev_cluster_idx_, reduce_clt_idx, -1, num_vtx_);
            CUDA_ERR_CHK(cudaDeviceSynchronize());
            CUDA_ERR_CHK(cudaPeekAtLastError());
            reduce_count++;
            clt_sem_vector.pop_back();
        }
    }
    CUDA_ERR_CHK(cudaMemcpy(cluster_ids, dev_cluster_idx_, N, D2H));
    CUDA_ERR_CHK(cudaFree(dev_cluster_idx_));
    return cluster_start + num_cluster-reduce_count;
}

void BINARY::Solver::assigned_LPs(int num_cluster, int cluster_start){

    const auto N = sizeof(x_[0]) * num_vtx_;

    int *point_num_clt;
    point_num_clt = (int *)malloc(sizeof(int) *num_cluster);

    // ===========================calculate the point number of cluster=============
    int cluster_id = 0;
    int noise_num = 0;
    int un_noise_num = 0;
    // init point number of cluster
    for (int i=0; i < num_cluster; i++){
        point_num_clt[i] =0;
    }
    // calculate the number
    for(int j=0; j<num_vtx_; j++){
        cluster_id = cluster_ids[j];
        if (cluster_id == -1){
            noise_num++;
        }
        else{
            point_num_clt[cluster_id-cluster_start]++;   //consider batch effect
        }
    }
    un_noise_num = num_vtx_ - noise_num;
    if(noise_num == 0) return;

    // ===========================check the index of noise=============
    int *noise_idx, *clt_idx;
    int noise_count =0;
    int un_noise_count =0;
    noise_idx = (int *)malloc(sizeof(int) * noise_num);
    clt_idx = (int *)malloc(sizeof(int) * un_noise_num);

    for(int k=0; k<num_vtx_; k++){
        if (cluster_ids[k]!=-1) {
            clt_idx[un_noise_count] = k;
            un_noise_count++;
        }
        else{
            noise_idx[noise_count] = k;
            noise_count++;
        }
    }
    // ===========================Assign=========================
    int k = 1; // only support k=1

    int *noise_idx_d, *clt_idx_d;
    CUDA_ERR_CHK( cudaMalloc((void **) &noise_idx_d, sizeof(int) * noise_num));
    CUDA_ERR_CHK( cudaMalloc((void **) &clt_idx_d, sizeof(int) * un_noise_num));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_cluster_idx_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_sem_, N));

    CUDA_ERR_CHK(cudaMemcpy(noise_idx_d, noise_idx, sizeof(int) * noise_num, H2D));
    CUDA_ERR_CHK(cudaMemcpy(clt_idx_d, clt_idx, sizeof(int) * un_noise_num, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_cluster_idx_, cluster_ids, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_sem_, sem_, N, H2D));

    // Memcpy from host to device
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_xo_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_yo_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_zo_, N));
    CUDA_ERR_CHK(cudaMemcpy(dev_xo_, xo_, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_yo_, yo_, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_zo_, zo_, N, H2D));

    // find the nearest point and cluster
    dim3 block_n(DIVUP(noise_num, THREADS_PER_BLOCK));
    dim3 threads_n(THREADS_PER_BLOCK);

    noise_id_cluster<<<block_n, threads_n>>>(dev_xo_, dev_yo_, dev_zo_,noise_idx_d,clt_idx_d,dev_cluster_idx_,
                                             dev_sem_, noise_num, un_noise_num);
    CUDA_ERR_CHK(cudaPeekAtLastError());

    CUDA_ERR_CHK(cudaMemcpy(cluster_ids, dev_cluster_idx_, N, D2H));

    CUDA_ERR_CHK(cudaFree(noise_idx_d));
    CUDA_ERR_CHK(cudaFree(clt_idx_d));
    CUDA_ERR_CHK(cudaFree(dev_cluster_idx_));
    CUDA_ERR_CHK(cudaFree(dev_xo_));
    CUDA_ERR_CHK(cudaFree(dev_yo_));
    CUDA_ERR_CHK(cudaFree(dev_zo_));
    CUDA_ERR_CHK(cudaFree(dev_sem_));

    free(point_num_clt);
    free(noise_idx);
    free(clt_idx);
}

void BINARY::Solver::get_clt_center(int num_cluster, int cluster_start, vector<float>& mean_vector){
    dim3 blocks(DIVUP(num_cluster, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    const auto N = sizeof(x_[0]) * num_vtx_;
    const auto M = sizeof(x_[0]) * num_cluster;

    // Malloc for host pointer
    float *mean_x, *mean_y, *mean_z;
    mean_x = (float *)malloc(M);
    mean_y = (float *)malloc(M);
    mean_z = (float *)malloc(M);

    // Malloc device pointer
    float *mean_xd, *mean_yd, *mean_zd;
    CUDA_ERR_CHK( cudaMalloc((void **) &mean_xd, M));
    CUDA_ERR_CHK( cudaMalloc((void **) &mean_yd, M));
    CUDA_ERR_CHK( cudaMalloc((void **) &mean_zd, M));

    CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_z_, N));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_cluster_idx_, N));

    // Memcpy from host to device
    CUDA_ERR_CHK(cudaMemcpy(dev_x_, x_, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_y_, y_, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_z_, z_, N, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_cluster_idx_, cluster_ids, N, H2D));

    //=============================calculate mean ====================================
    cal_mean<<<blocks, threads>>>(mean_xd,  mean_yd, mean_zd, dev_x_, dev_y_, dev_z_, dev_cluster_idx_, num_cluster,
                                  num_vtx_, cluster_start);

    CUDA_ERR_CHK(cudaDeviceSynchronize());
    CUDA_ERR_CHK(cudaPeekAtLastError());

    // Memcpy from device to host
    CUDA_ERR_CHK(cudaMemcpy(mean_x, mean_xd, M, D2H));
    CUDA_ERR_CHK(cudaMemcpy(mean_y, mean_yd, M, D2H));
    CUDA_ERR_CHK(cudaMemcpy(mean_z, mean_zd, M, D2H));
    // cuda free
    CUDA_ERR_CHK(cudaFree(mean_xd));
    CUDA_ERR_CHK(cudaFree(mean_yd));
    CUDA_ERR_CHK(cudaFree(mean_zd));
    CUDA_ERR_CHK(cudaFree(dev_x_));
    CUDA_ERR_CHK(cudaFree(dev_y_));
    CUDA_ERR_CHK(cudaFree(dev_z_));

    for(int i=0; i<num_cluster; i++){
        mean_vector.push_back(mean_x[i]);
        mean_vector.push_back(mean_y[i]);
        mean_vector.push_back(mean_z[i]);
    }

}
