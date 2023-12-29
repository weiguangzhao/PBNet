# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/28  下午10:35
# File Name: config.py
# IDE: PyCharm

import argparse

# config para
def get_parser():

    # # Train config
    parser = argparse.ArgumentParser(description='3D instance segmentation')
    parser.add_argument('--task', type=str, default='train', help='task: train or test')
    parser.add_argument('--manual_seed', type=int, default=22, help='seed to produce')
    parser.add_argument('--epochs', type=int, default=520, help='Total epoch')
    parser.add_argument('--num_works', type=int, default=4, help='num_works for dataset')
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--save_freq', type=int, default=4, help='Pre-training model saving frequency(epoch)')
    parser.add_argument('--logpath', type=str, default='./log/config_1/', help='path to save logs')
    parser.add_argument('--cache', type=bool, default=True, help='Whether to use shm')
    parser.add_argument('--validation', type=bool, default=True, help='Whether to verify the validation set')

    # #Dataset setting
    parser.add_argument('--dataset', type=str, default='Scannet', help='datasets')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='voxel_size for voxelize')
    parser.add_argument('--scale_size', type=float, default=1, help='voxel_size for voxelize')
    parser.add_argument('--sem_num', type=int, default=20, help='kinds of the sematic label')
    parser.add_argument('--max_crop_p', type=int, default=300000, help='max point of scene for crop')
    parser.add_argument('--min_crop_p', type=int, default=50000, help='max point of scene for crop')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size for single GPU')
    parser.add_argument('--batch_size_v', type=int, default=1, help='batch_size for single GPU (validation)')

    # #Adjust learning rate
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--step_epoch', type=int, default=50, help='How many steps apart to decay the learning rate')
    parser.add_argument('--multiplier', type=float, default=0.5, help='Learning rate decay: lr = lr * multiplier')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay for SGD')

    # #PBNet Clustering Parameter
    parser.add_argument('--cluster_epoch', type=int, default=128, help='after cluster_epoch, cluster start')
    parser.add_argument('--min_pts', type=int, default=31, help='minimum point for cluster')
    parser.add_argument('--radius', type=float, default=0.04, help='cluster_radius for cluster')
    parser.add_argument('--method', type=int, default=0, help='noise point identify')
    # #score net
    parser.add_argument('--fg_thresh', type=float, default=0.95, help='')
    parser.add_argument('--bg_thresh', type=float, default=0.20, help='')
    parser.add_argument('--fg_score', type=float, default=-1.0, help='')

    # #test
    parser.add_argument('--TEST_NMS_THRESH', type=float, default=0.10, help='')
    parser.add_argument('--TEST_SCORE_THRESH', type=float, default=0.07, help='')
    parser.add_argument('--TEST_NPOINT_THRESH', type=int, default=101, help='')
    parser.add_argument('--test_epoch', type=int, default=392, help='')

    # #Distributed training
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('-nr', '--node_rank', type=int, default=0, help='ranking within the nodes')
    parser.add_argument('--nodes', type=int, default=1, help='Number of distributed training nodes')
    parser.add_argument('--gpu_per_node', type=int, default=3, help='Number of GPUs per node')
    parser.add_argument('--sync_bn', type=bool, default=True, help='Whether to batch norm all gpu para')
    parser.add_argument('--tcp_port', type=int, default=16677, help='tcp port for Distributed training')

    args = parser.parse_args()
    return args