# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/9/30  下午4:12
# File Name: pbnet_ops
# IDE: PyCharm

import torch
from torch.autograd import Function
import PB_lib


class Cluster(Function):
    @staticmethod
    def forward(ctx, ins_offseted, ins_orig, sem, ins_bp, radius, min_pts, batch_size):
        # #### offseted xyz
        x = ins_offseted[:, 0].type(torch.float32).contiguous()
        y = ins_offseted[:, 1].type(torch.float32).contiguous()
        z = ins_offseted[:, 2].type(torch.float32).contiguous()
        l1_norm = torch.abs(x) + torch.abs(y) + torch.abs(z)
        index_mapper_list = []
        for batch_i in range(batch_size):
            num_bp = ins_bp[batch_i]
            index_mapper = torch.arange(0, num_bp, 1)
            index_mapper_list.append(index_mapper)
        index_mapper = torch.cat(index_mapper_list, dim=0).type(torch.int32).contiguous()
        # ####original xyz
        xo = ins_orig[:, 0].type(torch.float32).contiguous()
        yo = ins_orig[:, 1].type(torch.float32).contiguous()
        zo = ins_orig[:, 2].type(torch.float32).contiguous()
        # ####semantic pred
        sem = sem.type(torch.int32).contiguous()
        # ####cluster parameter radius & min_pts for 18 category
        radius = torch.ones(18) * radius
        radius = radius.type(torch.float32).contiguous()
        min_pts = torch.ones(18) * min_pts
        min_pts = min_pts.type(torch.int32).contiguous()

        # ####points nums and cluster id init
        ins_point_num = ins_offseted.shape[0]
        cluster_id = torch.ones(ins_point_num) * -1
        cluster_id = cluster_id.type(torch.int32).contiguous()
        # ####clusters nums for each batch
        batch_size = ins_bp.shape[0]
        cluster_num = torch.zeros([batch_size]).type(torch.int32).contiguous()
        # ####density queue
        den_queue = torch.zeros(ins_point_num, dtype=torch.int32).contiguous()
        # ####init clusters center address
        center = torch.zeros(ins_point_num, dtype=torch.float32).contiguous()
        # ####init cluster semantic address
        clt_sem = torch.zeros(ins_point_num, dtype=torch.int32).contiguous()

        # ####double check contiguous for pointer(*)
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert z.is_contiguous()
        assert l1_norm.is_contiguous()
        assert index_mapper.is_contiguous()
        assert xo.is_contiguous()
        assert yo.is_contiguous()
        assert zo.is_contiguous()
        assert sem.is_contiguous()
        assert ins_bp.is_contiguous()
        assert cluster_id.is_contiguous()
        assert min_pts.is_contiguous()
        assert radius.is_contiguous()
        assert cluster_num.is_contiguous()
        assert den_queue.is_contiguous()
        assert center.is_contiguous()
        assert clt_sem.is_contiguous()
        para_f = 0.05
        nv_flag = True

        PB_lib.binary_cluster(x, y, z, l1_norm, index_mapper, xo, yo, zo, sem, ins_bp, radius, min_pts, cluster_id,
                              cluster_num, den_queue, center, clt_sem, batch_size, para_f, nv_flag)
        return cluster_id, cluster_num, den_queue + 1, center

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


cluster = Cluster.apply


class GetIoU(Function):
    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum):
        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        proposals_idx = proposals_idx.type(torch.int32)
        proposals_offset = proposals_offset.type(torch.int32)

        PB_lib.get_iou(proposals_idx, proposals_offset, instance_labels, instance_pointnum, proposals_iou, nInstance,
                        nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_iou = GetIoU.apply


class CalIoUAndMasklabel(Function):
    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum, mask_scores_sigmoid, mode):
        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()
        mask_label = torch.cuda.FloatTensor(mask_scores_sigmoid.shape).zero_() - 1.

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda
        assert mask_scores_sigmoid.is_contiguous() and mask_scores_sigmoid.is_cuda

        proposals_idx = proposals_idx.type(torch.int32)
        proposals_offset = proposals_offset.type(torch.int32)

        PB_lib.cal_iou_and_masklabel(proposals_idx, proposals_offset, instance_labels, instance_pointnum,
                                      proposals_iou, nInstance, nProposal, mask_scores_sigmoid, mask_label, mode)

        return proposals_iou, mask_label

    @staticmethod
    def backward(ctx, a=None):
        return None, None


cal_iou_and_masklabel = CalIoUAndMasklabel.apply


class Get_normal_line(Function):
    @staticmethod
    def forward(ctx, xyz, face):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :return: proposals_iou: (nProposal, total_nInst), float
        '''
        xyz = torch.from_numpy(xyz).type(torch.float32).contiguous()
        face = torch.from_numpy(face).type(torch.int32).contiguous()
        normal_line = torch.zeros_like(xyz).type(torch.float32).contiguous()

        assert xyz.is_contiguous()
        assert face.is_contiguous()
        assert normal_line.is_contiguous()

        PB_lib.cal_normal_line(xyz, face, normal_line, xyz.shape[0], normal_line.shape[0])

        return normal_line

    @staticmethod
    def backward(ctx, a=None):
        return None


get_normal_line = Get_normal_line.apply
