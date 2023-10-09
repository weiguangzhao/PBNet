#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "PB_lib.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("binary_cluster", &binary_cluster, "binary_cluster");
    m.def("get_iou", &get_iou, "get_iou");
    m.def("cal_iou_and_masklabel", &cal_iou_and_masklabel, "cal_iou_and_masklabel");
    m.def("cal_normal_line", &cal_normal_line, "cal_normal_line");
}