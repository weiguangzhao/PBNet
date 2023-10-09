# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/10/21  下午4:45
# File Name: evaluate.py
# IDE: PyCharm

import os
import time
import torch
import numpy as np
import random
from tensorboardX import SummaryWriter

from config.config_test import get_parser
from datasets.scannetv2.dataset_preprocess import Dataset
import tools.log as log
from tools.mIOU import non_max_suppression
import tools.eval as eval
from tools.getins import align_superpoint_label

def init():
    global cfg
    cfg = get_parser()
    cfg.task = 'test'
    cfg.dist = False

    global result_dir
    result_dir = os.path.join('result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), "val")
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def eval_epoch(val_loader, model, model_fn_eval, epoch):
    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    with torch.no_grad():
        model.eval()
        gt_dir = 'datasets/scannetv2'

        matches = {}
        for i, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            pred = model_fn_eval(batch, model, epoch, cfg, task='test')
            # #==========================================sem eval=========================================
            pred_sem = pred['sem']

            # #==========================================ins eval=========================================
            if (epoch > cfg.cluster_epoch):
                val_scene_name = batch['fn'][0]
                superpoint = batch['sup']
                superpoint = torch.from_numpy(superpoint)
                point_num = batch['xyz_original'].shape[0]
                proposals_idx, proposals_offset, clt_score_v, proposals_ms = pred['proposals']
                clt_score = pred['clt_scores'].view(-1)

                semantic_id = torch.tensor(semantic_label_idx, device=torch.cuda.current_device())
                test = pred_sem[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]
                semantic_id = semantic_id[test]

                proposals_idx[:, 1] = proposals_idx[:, 1]%(point_num/3)
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, point_num//3), dtype=torch.int,
                                             device=clt_score.device)  # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1


                # # #### score threshold
                score_mask = (clt_score > cfg.TEST_SCORE_THRESH)
                clt_score = clt_score[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                # # #### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                clt_score = clt_score[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

                # ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f,
                                            proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), clt_score.cpu().numpy(),
                                                    cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = clt_score[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]
                if clusters.shape[0] == 0:
                    print('no cluster')
                    continue
                #
                seg_result = torch.ones(point_num//3) * -100
                for c_i in range(clusters.shape[0]):
                    cur_idx = torch.nonzero(clusters[c_i, :] == 1).view(-1)
                    seg_result[cur_idx] = c_i
                seg_result = seg_result.type(torch.int64).cuda()
                sp_labels, sp_scores = align_superpoint_label(seg_result, superpoint, clusters.shape[0])
                seg_result = sp_labels[superpoint]

                clusters[:, :] = 0
                pick_idxs = [p_i for p_i in range(clusters.shape[0])]
                for c_i in range(clusters.shape[0]):
                    cur_idx = torch.nonzero(seg_result == c_i).view(-1)
                    if cur_idx.shape[0]==0:
                        pick_idxs.remove(c_i)
                    clusters[c_i, cur_idx] = 1
                clusters = clusters[pick_idxs]
                cluster_scores = cluster_scores[pick_idxs]
                cluster_semantic_id = cluster_semantic_id[pick_idxs]

                # ####full time
                time_end = time.time()
                nclusters = clusters.shape[0]
                #
                ##### prepare for evaluation
                pred_info = {}
                pred_info['conf'] = cluster_scores.cpu().numpy()
                pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                pred_info['mask'] = clusters.cpu().numpy()

                gt_file = os.path.join(gt_dir, 'val_gt', val_scene_name + '.txt')
                gt2pred, pred2gt = eval.assign_instances_for_scan(val_scene_name, pred_info, gt_file)
                matches[val_scene_name] = {}
                matches[val_scene_name]['gt'] = gt2pred
                matches[val_scene_name]['pred'] = pred2gt

                print("complete {}, has {} clts".format(i, nclusters))
                # ##### write the txt file for submitted
                # f = open(os.path.join(result_dir, val_scene_name + '.txt'), 'w')
                # for proposal_id in range(nclusters):
                #     clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                #     semantic_label = np.argmax(np.bincount(pred_sem[np.where(clusters_i == 1)[0]].cpu()))
                #     score = cluster_scores[proposal_id]
                #     f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(val_scene_name, proposal_id,
                #                                                              semantic_label_idx[semantic_label], score))
                #     if proposal_id < nclusters - 1:
                #         f.write('\n')
                #     np.savetxt(
                #         os.path.join(result_dir, 'predicted_masks', val_scene_name + '_%03d.txt' % (proposal_id)),
                #         clusters_i, fmt='%d')
                # f.close()
        ap_scores = eval.evaluate_matches(matches)
        avgs = eval.compute_averages(ap_scores)
        eval.print_results(avgs, logger)


def Single_card_testing(gpu, cfg):
    # #logger
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # #summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    # #create model
    logger.info('=> creating model ...')
    from network.PBNet import PBNet as net
    from network.PBNet import model_fn_eval
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    model = net(cfg)
    model = model.to(gpu)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    # load dataset
    dataset = Dataset(cfg)
    logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, None, cfg.logpath, pretrain_file=cfg.pretrain, gpu=gpu)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))

    # dataset.testLoader()
    # eval_epoch(dataset.test_data_loader, model, model_fn_test, start_epoch)

    dataset.valLoader()
    eval_epoch(dataset.val_data_loader, model, model_fn_eval, start_epoch)
    pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    init()
    cfg.world_size = cfg.nodes * cfg.gpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    Single_card_testing(cfg.local_rank, cfg)


