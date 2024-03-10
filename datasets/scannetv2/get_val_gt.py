"""Generate instance groundtruth .txt files (for evaluation)"""

import glob
import numpy as np
import os
import torch

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
    'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
]

if __name__ == '__main__':
    split = 'val'
    SCANNET_DIR = './datasets/scannetv2/'
    files = np.loadtxt(SCANNET_DIR + 'scannetv2_{}.txt'.format(split), dtype='str')
    os.makedirs(SCANNET_DIR + split + '_gt', exist_ok=True)

    for i in range(len(files)):
        label = np.load(SCANNET_DIR + 'npy/' + files[i] + '_sem_label.npy')
        instance_label = np.load(SCANNET_DIR + 'npy/' + files[i] + '_ins_label.npy')
        scene_name = files[i]
        print('{}/{} {}'.format(i + 1, len(files), scene_name))

        instance_label_new = np.zeros(
            instance_label.shape,
            dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if (sem_id == -100):
                sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(SCANNET_DIR + split + '_gt', scene_name + '.txt'), instance_label_new, fmt='%d')