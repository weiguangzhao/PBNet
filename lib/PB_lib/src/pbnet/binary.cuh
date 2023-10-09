/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午7:34
# File Name: binary
# IDE: CLion
*/
#ifndef PBNET_BINARY_CUH
#define PBNET_BINARY_CUH

#include <thrust/execution_policy.h>  
#include <thrust/scan.h>
#include <vector>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

// https://stackoverflow.com/a/14038590 to check the cuda status
#define CUDA_ERR_CHK(code) { cuda_err_chk((code), __FILE__, __LINE__); }

inline void cuda_err_chk(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "\tCUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


namespace BINARY {
int const BLOCK_SIZE = 512;

class Solver {
 public:
   Solver(float *, float *, float *, float *,  int *, float *, float *, float *, int *, int, float *, int *, int*, int *);

  //====Sort the input by the l1norm of each point.
  void sort_input_by_l1norm();

  //====calculate the number of neighbor
  void calc_num_neighbours();

  //====Prefix sum.
  void calc_start_pos();

  //====populate the actual neighbours for each vertex.
  void append_neighbours();

  //===== Identify HPs
  void identify_HPs();

  //==== Cluster HPs
  int identify_clusters(int);

  //====filter fragment
  int filter(int, int,float, std::vector<int>&);

  //==== assigned LPs to clusters
  void assigned_LPs(int, int);

  //====get clusters' center coords
  void get_clt_center(int, int, std::vector<float>& );

 public:
  int *cluster_ids;
  int *memberships;

 private:
  cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;
  cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
  // query params
  int num_vtx_{};
  int total_num_nbs_{};
  // data structures
  float *x_{}, *y_{}, *z_{}, *l1norm_{};
  float *xo_{}, *yo_{}, *zo_{};
  int *sem_{};
  int *min_pts_{};
  float *radius_{};
  int *den_queue_{};
  // maps the sorted indices of each vertex to the original index.
  int *vtx_mapper_{};
  // gpu vars. Class members to avoid unnecessary copy.
  int num_blocks_{};
  float *dev_x_{}, *dev_y_{}, *dev_z_{}, *dev_l1norm_{};
  float *dev_xo_{}, *dev_yo_{}, *dev_zo_{};
  int *dev_vtx_mapper_{}, *dev_num_neighbours_{}, *dev_start_pos_{}, *dev_neighbours_{};
  int *dev_membership_{};
  int *dev_cluster_idx_{};
  int *dev_sem_{};
  int *dev_min_pts_{};
  float *dev_radius_{};

  void bfs_sem(int u, int cluster);
};
}

#endif