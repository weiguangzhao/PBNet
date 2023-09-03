# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/7/31  下午8:39
# File Name: mIOU.py
# IDE: PyCharm
import torch
import numpy as np

sem_class = {'floor': 0, 'wall': 1, 'cabinet': 2, 'bed': 3, 'chair': 4, 'sofa': 5, 'table': 6, 'door': 7,
             'window': 8, 'bookshelf': 9, 'picture': 10, 'counter': 11, 'desk': 12, 'curtain': 13,
             'refrigerator': 14, 'showercurtrain': 15, 'toilet': 16, 'sink': 17, 'bathtub': 18, 'otherfurniture': 19}

Ins_class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
             'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
             'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'otherfurniture': 17}


def intersectionAndUnionGPU(output, target, K, ignore_index=-100):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
    '''
    :param scores: (N), float, 0~1
    :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    '''
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores


def get_center_scores(dist, near_thresh=0.3, far_thresh=0.6):
    '''
    :param scores: (N), float, 0~1
    :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    '''
    far_mask = dist > far_thresh
    near_mask = dist < near_thresh
    interval_mask = (far_mask == 0) & (near_mask == 0)

    center_scores = (near_mask > 0).float()
    k = 1 / (far_thresh - near_thresh)
    b = far_thresh / (far_thresh - near_thresh)
    center_scores[interval_mask] = b - dist[interval_mask] * k

    return center_scores


def get_gt_dist(instance_info, ins_label, gt_instance_idxs, pred_center):
    valid_index = torch.nonzero(ins_label != -100).view(-1)
    gt_center = torch.cat((ins_label.unsqueeze(-1), instance_info[:, :3]), dim=1)
    gt_center = torch.unique(gt_center[valid_index, ...], dim=0, sorted=True)
    gt_center = gt_center[gt_instance_idxs, ...][:, 1:]
    gt_dist = torch.norm((gt_center-pred_center), dim=1, p=2).detach()
    return gt_dist


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
