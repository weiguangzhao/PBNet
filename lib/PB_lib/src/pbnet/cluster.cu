/*
#_*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/29  下午7:34
# File Name: cluster
# IDE: CLion
*/
#include <fstream>
#include <vector>
#include <algorithm>

#include "binary.cuh"
#include "cluster.h"

using namespace std;
void binary_cluster(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor l1_norm_tensor,
                    at::Tensor index_mapper_tensor, at::Tensor xo_tensor, at::Tensor yo_tensor, at::Tensor zo_tensor,
                    at::Tensor sem_tensor, at::Tensor batch_index_tensor, at::Tensor radius_tensor,
                    at::Tensor min_pts_tensor, at::Tensor cluster_index_tensor, at::Tensor cluster_num_tensor,
                    at::Tensor den_queue_tensor, at::Tensor center_tensor, at::Tensor clt_sem_tensor, int batch_size,
                    float para_f, bool nv_flag) {

    // ======================================get C++ pointer=============================
    //=====for kd tree  (offset xyz)
    float *x = x_tensor.data<float>();
    float *y = y_tensor.data<float>();
    float *z = z_tensor.data<float>();
    float *l1_norm = l1_norm_tensor.data<float>();
    int *index_mapper = index_mapper_tensor.data<int>();
    //====original xyz
    float *xo = xo_tensor.data<float>();
    float *yo = yo_tensor.data<float>();
    float *zo = zo_tensor.data<float>();
    //====cluster information
    int *sem = sem_tensor.data<int>();                  //semantic info for each point
    int *batch_ind = batch_index_tensor.data<int>();     //batch division
    float *radius = radius_tensor.data<float>();
    int *min_pts = min_pts_tensor.data<int>();
    //=====what we want to get
    int *cluster_idx = cluster_index_tensor.data<int>(); //cluster index for points
    int *cluster_num = cluster_num_tensor.data<int>();  //cluster number for batch
    int *den_queue = den_queue_tensor.data<int>();       // density for each points
    //====pointers for batch variables
    float *x_s, *y_s, *z_s, *l1_norm_s;
    float *xo_s, *yo_s, *zo_s;
    int *index_mapper_s, *cluster_idx_s, *sem_s, *den_queue_s;

    int batch_length = 0;
    int batch_start = 0;
    int cluster_accum = 0;
    int cluster_accum_old = 0;
    //====vector for cluster center
    vector<float> mean_vector;
    //====vector for cluster semantic
    vector<int> clt_sem_vector;
    // ======================================cluster the points for each batch=============================
    for(int batch_i=0; batch_i<batch_size; batch_i++){
        batch_length = batch_ind[batch_i];
        if (batch_length==0) {
            continue;
        }
        // =====pointer offset for batch
        x_s = x + batch_start;
        y_s = y + batch_start;
        z_s = z + batch_start;
        l1_norm_s = l1_norm + batch_start;
        index_mapper_s = index_mapper + batch_start;

        xo_s = xo + batch_start;
        yo_s = yo + batch_start;
        zo_s = zo + batch_start;
        sem_s = sem + batch_start;
        cluster_idx_s = cluster_idx + batch_start;
        den_queue_s = den_queue + batch_start;

        //==================Binary cluster==========================
        //====Public variables
        BINARY::Solver solver(x_s, y_s, z_s, l1_norm_s, index_mapper_s, xo_s, yo_s, zo_s, sem_s, batch_length, radius,
                               min_pts, cluster_idx_s, den_queue_s);
        //====Sort the input by the l1norm of each point.
        solver.sort_input_by_l1norm();
        //====calculate the number of neighbor
        solver.calc_num_neighbours();
        //====Prefix sum.
        solver.calc_start_pos();
        //====populate the actual neighbours for each vertex.
        solver.append_neighbours();
        //===== identify HPs
        solver.identify_HPs();
        //==== Cluster HPs
        cluster_accum = solver.identify_clusters(cluster_accum_old);
        //==== change  fragments to be LPs
        cluster_accum = solver.filter(cluster_accum-cluster_accum_old, cluster_accum_old, para_f, clt_sem_vector);
        //==== assigned LPs to clusters
        if(nv_flag) solver.assigned_LPs(cluster_accum-cluster_accum_old, cluster_accum_old);

        cluster_num[batch_i] = cluster_accum - cluster_accum_old;
        if(cluster_num[batch_i] == 0) {
            //printf("batch %d has none cluster \n", batch_i);
            cluster_accum_old = cluster_accum;
            batch_start += batch_length;
            continue;
        }
        solver.get_clt_center(cluster_num[batch_i], cluster_accum_old, mean_vector);


        //====refresh==============
        cluster_accum_old = cluster_accum;
        batch_start += batch_length;
    }
    //========================output resize=======================
    center_tensor.resize_({((int)mean_vector.size())});
    float *center_xyz = center_tensor.data<float>();
    memcpy(center_xyz, &mean_vector[0], mean_vector.size()*sizeof(float ));

    clt_sem_tensor.resize_({((int)clt_sem_vector.size())});
    int *clt_sem = clt_sem_tensor.data<int>();
    memcpy(clt_sem, &clt_sem_vector[0], clt_sem_vector.size()*sizeof(int ));
}