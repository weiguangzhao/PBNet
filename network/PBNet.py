# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/29  下午3:08
# File Name: PBNet.py
# IDE: PyCharm

import torch
import torch.nn as nn

import random
import MinkowskiEngine as ME

from tools.mIOU import get_segmented_scores
from network.Mink import Mink_unet as unet3d
from lib.PB_lib.torch_io import  pbnet_ops
from tools.plt import get_ptcloud_img, get_ptcloud_img_v2, get_ptcloud_img_v3

class PBNet(nn.Module):
    def __init__(self, cfg):
        super(PBNet, self).__init__()
        # # config
        self.batch_size = cfg.batch_size
        self.cluster_batch = cfg.batch_size * 1
        self.sem_num = cfg.sem_num
        self.voxel_size = cfg.voxel_size
        self.scale_size = cfg.scale_size

        self.cluster_epoch = cfg.cluster_epoch
        self.radius = cfg.radius
        self.min_pts = cfg.min_pts
        self.method = cfg.method
        # count mean from softgroup & HAIS
        self.count_mean = torch.tensor([-1., -1., 3917., 12056., 2303., 8331., 3948., 3166., 5629., 11719., 1003.,
                                        3317., 4912., 10221., 3889., 4136., 2120., 945., 3967., 2589.])
        self.K_max = torch.ones(20, dtype=torch.float32) * 6

        # # ME-UNet
        self.MEUnet = unet3d(in_channels=6, out_channels=32, arch='MinkUNet34C')
        self.D_Unet = unet3d(in_channels=34, out_channels=32, arch='MinkUNet14A')    # ###local scene
        self.score_Unet = unet3d(in_channels=32, out_channels=32, arch='MinkUNet34C') # ####score net

        # ####sematic
        self.linear_sem = nn.Sequential(
            ME.MinkowskiLinear(32, 16, bias=False),
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
        # ####mask
        self.linear_binary = nn.Sequential(
            ME.MinkowskiLinear(32, 16, bias=False),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(16, 1, bias=True),
            ME.MinkowskiSigmoid()
        )

        # ###score
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear_IOU_feat = nn.Sequential(
            ME.MinkowskiLinear(32, 32, bias=False),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(32, 32, bias=True)
        )

        self.linear_IOU = nn.Sequential(
            ME.MinkowskiLinear(32, 16, bias=False),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiPReLU(),
            ME.MinkowskiLinear(16, 1, bias=True),
            ME.MinkowskiSigmoid()
        )

        # ####attention global feat
        self.soft_max = ME.MinkowskiSoftmax()

        # #####init weight
        self.weight_initialization()

        # #####frozen net to reduce room
        self.fix_module = []
        # self.fix_module = ['Unet_backbone', 'linear_sem', 'linear_off', 'D_Unet'] # #for train score net

        module_map = {'Unet_backbone': self.MEUnet,
                      'linear_sem': self.linear_sem,
                      'linear_off': self.linear_offset,
                      'D_Unet': self.D_Unet}
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

    def forward(self, feat_voxel, xyz_voxel, xyz_original, v2p_v1, ins_label, epoch, task='train'):
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
        batch_head_p = xyz_voxel[:, 0][v2p_v1, ...]  # [N] int32   (get batch head for point)
        # #####semantic pred
        sem_pred_p = sem_pred_score_p.max(1)[1]  # [N] int64
        # #####return backbone result
        ret = {}
        ret['sem_pred_p'] = sem_pred_p  # [N, 1]  int64
        ret['sem_pred_score_p'] = sem_pred_score_p  # [N, 20] float32
        ret['offset_pred_p'] = offset_pred_p  # [N, 3]  float32

        # #==========================cluster stage===============================================
        if epoch > self.cluster_epoch:
            # ####package instance info
            list_xyz = []
            list_feat = []
            list_gt_mask = []
            list_ins_idx = []
            # ####handle each semantic
            for sem_id in range(int(self.sem_num - 2)):
                sem_id = sem_id + 2
                # ####get the instance index for current semantic
                ins_ind = torch.nonzero(sem_pred_p == sem_id).view(-1)
                ins_ind = torch.sort(ins_ind)[0]
                if ins_ind.shape[0] < self.count_mean[sem_id] * 0.05: continue
                # ####current instance info
                ins_orig = xyz_original[ins_ind, ...]  # [I, 3]  float32
                ins_offset = offset_pred_p[ins_ind, ...]  # [I, 3]  float32
                ins_feat = point_feat_p[ins_ind, ...]
                ins_sem = sem_pred_p[ins_ind, ...]
                cur_sem_score = sem_pred_score_sfp[:, sem_id]
                ins_sem_score = cur_sem_score[ins_ind, ...]
                if task != 'test': ins_ins_label = ins_label[ins_ind]
                ins_offseted = ins_orig.cpu() + ins_offset.cpu()
                # ####batch division (single num/ sum num)
                if task=='train':
                    self.cluster_batch = self.batch_size
                else:
                    self.cluster_batch = 3
                ins_bh = batch_head_p[ins_ind]
                ins_bp_db = self.get_batch_offset(ins_bh)
                ins_bp_sum = self.get_batch_offsets_sum(ins_bh, self.cluster_batch)

                # ####use cuda to cluster
                cluster_id, cluster_num, den_queue, clt_ctr = pbnet_ops.cluster(ins_offseted, ins_orig.cpu(),
                                                                             ins_sem.cpu(),
                                                                             ins_bp_db.cpu(), self.radius, self.min_pts,
                                                                             self.cluster_batch)
                # #=================================construct local scene==================================
                # ####cluster center handle
                clt_ctr = clt_ctr.view(-1, 3)
                ctr_offset = self.get_center_index_sum(cluster_num, self.cluster_batch)
                # ####batch division
                for cur_bi in range(self.cluster_batch):
                    if cluster_num[cur_bi] == 0: continue
                    # ####batch info
                    batch_xyz_orig = ins_orig[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1], ...]
                    batch_feat = ins_feat[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1], ...]
                    batch_sem_sf = ins_sem_score[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1], ...].view(-1, 1)
                    batch_ins_idx = ins_ind[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1]]
                    batch_clt_id = cluster_id[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1], ...]
                    if task != 'test': batch_ins_label = ins_ins_label[ins_bp_sum[cur_bi]:ins_bp_sum[cur_bi + 1]]                    # ####point feat concat semantic score(softmax)
                    batch_ins_feat = torch.cat((batch_feat.cpu(), batch_sem_sf.cpu()), dim=1)
                    # ####search instance for local scene
                    para_k = min((cluster_num[cur_bi] - 1, int(self.K_max[sem_id])))
                    if para_k > 0:
                        peak_v = [0.5 * ((para_k + 1) - p_i) / (para_k + 1) for p_i in range(para_k + 1)]
                        clt_center = clt_ctr[ctr_offset[cur_bi]:ctr_offset[cur_bi + 1], ...]
                        dist = torch.cdist(clt_center, clt_center)
                        knn_idx = dist.topk(k=cluster_num[cur_bi], dim=1, largest=False)[1].cpu()
                    # ####generate proposal mask for local scene
                    for c_i in range(cluster_num[cur_bi]):
                        valid_idx = torch.nonzero(batch_clt_id == c_i + ctr_offset[cur_bi]).view(-1)
                        if task != 'test': cur_gt_ins_label = torch.mode(batch_ins_label[valid_idx].cpu())[0]
                        if task != 'test' and cur_gt_ins_label == -100: continue
                        cur_dpn = torch.ones(valid_idx.shape[0])
                        # ####local scene is only used for middle and large instance
                        if valid_idx.shape[0] > self.count_mean[sem_id] * 0.2 and para_k > 0:
                            sub_valid_list = []
                            sub_dpn_list = []
                            sub_valid_list.append(valid_idx)
                            sub_dpn_list.append(cur_dpn)
                            for k_i in range(para_k):
                                valid_idx = torch.nonzero(
                                    batch_clt_id.cpu() == knn_idx[c_i, k_i + 1] + ctr_offset[cur_bi]).view(-1)
                                cur_dpn = torch.ones(valid_idx.shape[0]) * peak_v[k_i]
                                sub_valid_list.append(valid_idx)
                                sub_dpn_list.append(cur_dpn)
                            valid_idx = torch.cat(sub_valid_list, dim=0)
                            cur_dpn = torch.cat(sub_dpn_list, dim=0)
                        # ####get GT mask for training
                        if task != 'test':
                            valid_ins_label = batch_ins_label[valid_idx]
                            cur_gt_mask = (valid_ins_label == cur_gt_ins_label).long()
                            ignore_idx = torch.nonzero(valid_ins_label == -100).view(-1)
                            cur_gt_mask[ignore_idx] = -1
                        assert cur_dpn.min() > 0.0
                        # ####add current instance or local scene
                        cur_feat = torch.cat((batch_ins_feat[valid_idx, ...], cur_dpn.view(-1, 1)), dim=1)
                        list_xyz.append(batch_xyz_orig[valid_idx])
                        list_feat.append(cur_feat)
                        if task != 'test': list_gt_mask.append(cur_gt_mask)
                        list_ins_idx.append(batch_ins_idx[valid_idx])
            # ####voxel proposal
            list_clt_xyz = [sem_xyz / 0.02 for sem_xyz in list_xyz]
            coords = ME.utils.batched_coordinates(list_clt_xyz)
            sem_feat_tensor = torch.cat(list_feat, dim=0)
            if task != 'test': gt_mask = torch.cat(list_gt_mask, dim=0)
            inputs_v2 = ME.SparseTensor(
                features=sem_feat_tensor,
                coordinates=coords,
                # quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                # minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=cuda_cur_device,
            )
            v2p_v2 = inputs_v2.inverse_mapping
            each_sem_feat = self.D_Unet(inputs_v2)
            mask_score = self.linear_binary(each_sem_feat)
            mask_score = mask_score.F[v2p_v2]
            if task != 'test': ret['mask_scores'] = (mask_score, gt_mask.detach().cuda())
            ret['proposals'] = self.get_proposal(list_ins_idx, mask_score)
            #
            # # # # ####==================================get filter shape========================
            proposals_idx, proposals_offset, _, each_sem_feat_v = ret['proposals']
            proposals_idx = proposals_idx.type(torch.int64)
            proposals_offset = proposals_offset.type(torch.int64)

            # ####batch data
            clt_length = (proposals_offset[1:] - proposals_offset[:-1]).tolist()
            ins_orig_sort = xyz_original[proposals_idx[:, 1], ...] * self.scale_size / self.voxel_size
            ins_feat_sort = point_feat_p[proposals_idx[:, 1], ...].cpu()
            ins_orig_sp = torch.split(ins_orig_sort, clt_length, dim=0)  # list of torch tensor
            coords = ME.utils.batched_coordinates(ins_orig_sp)
            inputs_v3 = ME.SparseTensor(
                features=ins_feat_sort,
                coordinates=coords,
                # quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                # minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=point_feat_p.device,
            )
            IOU_feat = self.score_Unet(inputs_v3)
            IOU_feat = self.linear_IOU_feat(IOU_feat)
            avg_feat = self.global_avg_pool(IOU_feat)
            max_feat = self.global_max_pool(IOU_feat)
            global_feat = max_feat + avg_feat
            clt_score = self.linear_IOU(global_feat)
            clt_score = clt_score.F
            ret['clt_scores'] = clt_score.view(-1)
        return ret

    def get_batch_offset(self, ins_bh):
        ins_bp = torch.zeros(self.cluster_batch).int().cuda()
        for i in range(self.cluster_batch):
            ins_bp[i] = (ins_bh == i).sum()
        assert ins_bp.sum() == ins_bh.shape[0]
        return ins_bp

    def get_batch_offsets_sum(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def get_center_index_sum(self, clt_num, bs):
        ctr_offset = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            ctr_offset[i + 1] = ctr_offset[i] + clt_num[i]
        assert ctr_offset[-1] == clt_num.sum()
        return ctr_offset.cpu()

    def get_label_mask(self, sem_ins_label, cur_ins_idx):
        cur_label = torch.mode(sem_ins_label[cur_ins_idx])[0]
        gt_mask = torch.zeros(sem_ins_label.shape[0], dtype=torch.int64)
        ign_ind = torch.nonzero(sem_ins_label == -100).view(-1)
        true_ind = torch.nonzero(sem_ins_label == cur_label).view(-1)
        if ign_ind is not None:
            gt_mask[ign_ind] = -1
        if true_ind is not None:
            gt_mask[true_ind] = 1
        if cur_label == -100:
            gt_mask[:] = -1
        print("ins_label_python:{}".format(cur_label))
        return gt_mask

    def get_proposal(self, list_idx_proposal, mask_score, mask_score_thd=0.45):
        proposals_idx = []
        assert torch.cat(list_idx_proposal, dim=0).shape[0] == mask_score.shape[0]
        for idx_i in range(len(list_idx_proposal)):
            cur_pro_idx = torch.ones([list_idx_proposal[idx_i].shape[0], 2])
            cur_pro_idx[:, 0] = cur_pro_idx[:, 0] * idx_i
            cur_pro_idx[:, 1] = list_idx_proposal[idx_i]
            proposals_idx.append(cur_pro_idx)
        proposals_idx = torch.cat(proposals_idx, dim=0).type(torch.int64)
        valid_index = torch.nonzero(mask_score.view(-1) > mask_score_thd).view(-1)
        proposals_idx = proposals_idx[valid_index]
        proposals_ms = mask_score[valid_index].view(-1)

        cluster_id_v, cluster_len = torch.unique(proposals_idx[:, 0], return_counts=True)
        cluster_id_v = torch.sort(cluster_id_v)[0]
        proposals_offset = torch.zeros(cluster_len.shape[0] + 1)
        for i in range(proposals_offset.shape[0]):
            if i == 0:
                continue
            proposals_offset[i] = torch.sum(cluster_len[:i])
        if proposals_offset.shape[0] == 1:
            return proposals_idx.type(torch.int64).detach(), proposals_offset.type(
                torch.int64).detach(), cluster_id_v.detach(), proposals_ms

        # #remove null proposals
        if cluster_id_v.shape[0] != torch.max(proposals_idx[:, 0]) + 1:
            for i in range(cluster_id_v.shape[0]):
                cor_idx = proposals_idx[:, 0] == cluster_id_v[i]
                proposals_idx[cor_idx, 0] = i
        return proposals_idx.type(torch.int64).detach(), proposals_offset.type(
            torch.int64).detach(), cluster_id_v.detach(), proposals_ms

def model_fn(batch, model, epoch,  cfg, task='train'):
    # #input para
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    xyz_original = batch['xyz_original']
    v2p_index = batch['v2p_index']
    ins_label = batch['ins'].cuda()

    ret = model(feat_voxel, xyz_voxel, xyz_original, v2p_index, ins_label, epoch, task)

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
    if epoch > cfg.cluster_epoch:
        pred_mask, gt_mask = ret['mask_scores']
        mask_label_weight = (gt_mask != -1).float()
        gt_mask[gt_mask == -1.] = 0.5  # any value is ok
        mask_criterion = nn.BCELoss(reduction='none', weight=mask_label_weight)
        mask_loss = mask_criterion(pred_mask.view(-1), gt_mask.type(torch.float32))
        mask_loss = mask_loss.mean()
        loss = loss + mask_loss

        dice_loss = diceLoss(pred_mask[gt_mask != -1].view(-1), gt_mask[gt_mask != -1].view(-1))
        loss += dice_loss

        # # # # # # #score loss
        proposals_idx, proposals_offset, _, _ = ret['proposals']  # proposals_train
        clt_scores_pred = ret['clt_scores']
        ious = pbnet_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), ins_label, instance_pointnum)
        gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
        gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)
        score_criterion = nn.BCELoss()
        score_loss = score_criterion(clt_scores_pred.view(-1), gt_scores)
        score_loss = score_loss.mean()
        loss = loss + score_loss

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

        if epoch > cfg.cluster_epoch:
            visual_dict['mask_loss'] = mask_loss.item()
            meter_dict['mask_loss'] = (mask_loss.item(), mask_label_weight.sum())
            pred['mask_scores'] = ret['mask_scores']
            pred['proposals'] = ret['proposals']
            pred['clt_scores'] = ret['clt_scores']
            pass

    return loss, pred, visual_dict, meter_dict

def model_fn_eval(batch, model, epoch,  cfg, task='test'):
    # #input para
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    xyz_original = batch['xyz_original']
    v2p_index = batch['v2p_index']
    ins_label = None
    ret = model(feat_voxel, xyz_voxel, xyz_original, v2p_index, ins_label, epoch, task)

    pred = {}
    pred['sem'] = ret['sem_pred_p']
    if epoch > cfg.cluster_epoch:
        pred['proposals'] = ret['proposals']
        pred['clt_scores'] = ret['clt_scores']
    return pred


def diceLoss(mask_pred, mask_gt, ep=1e-8):
    inter = 2 * (mask_gt * mask_pred).sum() + 1
    union = (mask_gt ** 2.0).sum() + (mask_pred ** 2.0).sum() + 1 + ep
    dice_loss = 1 - inter / union

    return dice_loss
