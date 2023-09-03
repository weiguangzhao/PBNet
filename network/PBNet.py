# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/29  下午3:08
# File Name: PBNet.py
# IDE: PyCharm

import torch
import torch.nn as nn

import random
import MinkowskiEngine as ME

from network.Mink import Mink_unet as unet3d



class PBNet(nn.Module):
    def __init__(self, cfg):
        super(PBNet, self).__init__()
        # # config
        self.batch_size = cfg.batch_size
        self.sem_num = cfg.sem_num
        self.voxel_size = cfg.voxel_size
        self.scale_size = cfg.scale_size

        # # ME-UNet
        self.MEUnet = unet3d(in_channels=6, out_channels=32, arch='MinkUNet34C')

        # ####sematic
        self.linear_sem = nn.Sequential(
            ME.MinkowskiLinear(32, 16, bias=False),
            # ME.MinkowskiDropout(),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(16, self.sem_num, bias=True)
        )

        # ####offset
        self.linear_offset = nn.Sequential(
            ME.MinkowskiLinear(32, 16, bias=False),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(16, 3, bias=True)
        )

        # ####attention global feat
        self.soft_max = ME.MinkowskiSoftmax()

        self.weight_initialization()
        self.fix_module = []

        module_map = {'Unet_backbone': self.MEUnet,
                      'linear_sem': self.linear_sem,
                      'linear_off': self.linear_offset}
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False
        pass

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, feat_voxel, xyz_voxel, xyz_original, v2p_v1, ins_label, sem_label, epoch, task='train'):
        # start_time = time.time()
        cuda_cur_device = torch.cuda.current_device()
        # #============================Unet V1===================================================================
        inputs_v1 = ME.SparseTensor(feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        # ####backbone
        point_feat = self.MEUnet(inputs_v1)
        # ####two branches
        sem_pred_score = self.linear_sem(point_feat)  # [V, 20] float32
        sem_pred_score_sf = self.soft_max(sem_pred_score)  # [V, 20] float32
        offsets_pred = self.linear_offset(point_feat)  # [V, 3] float32
        # ####change sparse tensor to torch tensor
        point_feat = point_feat.F  # backbone feature
        sem_pred_score = sem_pred_score.F  # from sparse tensor to torch tensor
        sem_pred_score_sf = sem_pred_score_sf.F  # from sparse tensor to torch tensor
        offsets_pred = offsets_pred.F  # from sparse tensor to torch tensor
        # #####Voxel to point
        point_feat_p = point_feat[v2p_v1, ...]  # [N, 32]
        sem_pred_score_p = sem_pred_score[v2p_v1, ...]  # [N, 20] float32
        sem_pred_score_sfp = sem_pred_score_sf[v2p_v1, ...]  # [N, 20]
        offset_pred_p = offsets_pred[v2p_v1, ...]  # [N, 3] float32
        # #####semantic pred
        sem_pred_p = sem_pred_score_p.max(1)[1]  # [N] int64
        # #####return backbone result
        ret = {}
        ret['sem_pred_p'] = sem_pred_p  # [N, 1]  int64
        ret['sem_pred_score_p'] = sem_pred_score_p  # [N, 20] float32
        ret['offset_pred_p'] = offset_pred_p  # [N, 3]  float32
        return ret

def model_fn(batch, model, epoch,  cfg, task='train'):
    # #input para
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    xyz_original = batch['xyz_original']
    v2p_index = batch['v2p_index']
    ins_label = batch['ins'].cuda()
    sem_label = batch['sem'].cuda()

    ret = model(feat_voxel, xyz_voxel, xyz_original, v2p_index, ins_label, sem_label, epoch, task)

    # #label
    sem_label = batch['sem'].cuda()
    ins_label = batch['ins'].cuda()
    instance_info = batch['inst_info'].cuda()
    instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
    xyz_original = xyz_original.cuda()

    # #============================calculate loss V1===============================================================
    sem_pred_score_p = ret['sem_pred_score_p']
    offset_pred_p = ret['offset_pred_p']
    sem_pred_p = ret['sem_pred_p']

    # # semantic loss_v1
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=-100).cuda()
    semantic_loss = semantic_criterion(sem_pred_score_p, sem_label)

    # # offset loss_v1: l1 distance
    # instance_info: (N, 9), float32 tensor (mean_xyz, min_xyz, max_xyz)
    # instance_labels: (N), long
    gt_offsets = instance_info[:, 0:3] - xyz_original  # [N, 3] float32  :get gt offset value
    pt_diff = offset_pred_p - gt_offsets  # [N, 3] float32  :l1 distance between gt and pred
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # [N]    float32  :sum l1
    valid = (ins_label != -100).float()  # # get valid num
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)  # # avg

    # # offset loss_v1: direction loss (cos)
    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # [N]    float32  :norm
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    pt_offsets_norm = torch.norm(offset_pred_p, p=2, dim=1)  # [N]    float32  :norm
    pt_offsets = offset_pred_p / (pt_offsets_norm.unsqueeze(-1) + 1e-8)  # [N, 3] float32  :unit vector
    direction_diff = - (gt_offsets_ * pt_offsets).sum(-1)  # [N]    float32  :direction diff (cos)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)  # # avg

    # # sum V1 loss
    loss = semantic_loss + offset_norm_loss + offset_dir_loss
    # #============================calculate Score loss===============================================================
    with torch.no_grad():
        pred = {}
        offseted_v1p = xyz_original + offset_pred_p
        pred['sem'] = sem_pred_p
        pred['offseted_xyz'] = offseted_v1p

        visual_dict = {}
        visual_dict['loss'] = loss.item()
        visual_dict['semantic_loss'] = semantic_loss.item()
        visual_dict['offset_norm_loss'] = offset_norm_loss.item()
        visual_dict['offset_dir_loss'] = offset_dir_loss.item()

        meter_dict = {}
        meter_dict['loss'] = (loss.item(), valid.sum())
        meter_dict['semantic_loss'] = (semantic_loss.item(), valid.sum())
        meter_dict['offset_norm_loss'] = (offset_norm_loss.item(), valid.sum())
        meter_dict['offset_dir_loss'] = (offset_dir_loss.item(), valid.sum())

    return loss, pred, visual_dict, meter_dict
