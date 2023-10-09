# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/29  下午1:50
# File Name: dataset_preprocess.py
# IDE: PyCharm

import math
import torch
import random
import numpy as np
import SharedArray as SA
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
import MinkowskiEngine as ME


class Dataset:
    def __init__(self, cfg):

        self.batch_size = cfg.batch_size
        self.batch_size_v = cfg.batch_size_v
        self.dataset_workers = cfg.num_works
        self.cache = cfg.cache
        self.dist = cfg.dist
        self.voxel_size = cfg.voxel_size
        self.scale_size = cfg.scale_size
        self.min_crop_p = cfg.min_crop_p
        self.mixup = True

        # #crop setting
        self.full_scale = [128*self.scale_size/50.0, 512*self.scale_size/50.0]
        self.max_crop_p = cfg.max_crop_p

        self.dataset_root = 'datasets'
        self.dataset = 'scannetv2'
        self.dataset_suffix = '.npy'
        self.npy_dir = 'datasets/scannetv2/npy/'

        # ####seprate the train val test set
        self.train_file_list = np.loadtxt('datasets/scannetv2/scannetv2_train.txt', dtype=str)
        self.val_file_list = np.loadtxt('datasets/scannetv2/scannetv2_val.txt', dtype=str)
        self.test_file_list = np.loadtxt('datasets/scannetv2/scannetv2_test.txt', dtype=str)
        self.train_file_list.sort()
        self.val_file_list.sort()
        self.test_file_list.sort()

    def trainLoader(self):
        train_set = list(range(len(self.train_file_list)))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge,
                                            num_workers=self.dataset_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
                                            drop_last=True, pin_memory=False,
                                            worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        val_set = list(range(len(self.val_file_list)))
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size_v, collate_fn=self.valMerge,
                                          num_workers=self.dataset_workers,
                                          shuffle=False, sampler=None, drop_last=False, pin_memory=True,
                                          worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        test_set = list(range(len(self.test_file_list)))
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_set) if self.dist else None
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None, drop_last=False, pin_memory=True,
                                           worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    def dataAugment(self, xyz, rgb, nl, i, jitter=False, flip=False, rot=False, scale=False, elastic=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi + math.pi * i *(2/3)
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        xyz = np.matmul(xyz, m)
        xyz = xyz - xyz.min(0)

        if scale and np.random.rand() < prob:
            scale_factor = np.random.uniform(0.95, 1.05)
            xyz = xyz * scale_factor

        if elastic and np.random.rand() < prob:
            xyz = self.elastic(xyz, 6, 40)
            xyz = self.elastic(xyz, 20, 160)
            xyz = xyz - xyz.min(0)

        # ####rgb
        rgb = rgb + np.random.randn(3) * 0.1
        return xyz, rgb, nl

    # # crop scenes
    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_crop_p):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32 * self.scale_size / 50.0

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def getInstLabel(self, instance_label):
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9),
                                dtype=np.float32) * -100.0  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def trainMerge(self, id):
        xyz_voxel = []
        feat_voxel = []

        xyz_original = []

        sem_batch = []
        ins_batch = []
        inst_info_batch = []
        v2p_index_batch = []
        instance_pointnum = []  # (total_nInst), int

        total_inst_num = 0
        total_voxel_num = 0

        file_name = []
        for i, idx in enumerate(id):
            fn = self.train_file_list[idx]  # get shm name
            if self.cache:
                xyz = SA.attach("shm://{}_xyz".format(fn)).copy()
                rgb = SA.attach("shm://{}_rgb".format(fn)).copy()
                sem = SA.attach("shm://{}_sem_label".format(fn)).copy()
                ins = SA.attach("shm://{}_ins_label".format(fn)).copy()
                nl = SA.attach("shm://{}_nl".format(fn)).copy()
            else:
                xyz = np.load(self.npy_dir + '{}_xyz.npy'.format(fn))
                rgb = np.load(self.npy_dir + '{}_rgb.npy'.format(fn))
                sem = np.load(self.npy_dir + '{}_sem_label.npy'.format(fn))
                ins = np.load(self.npy_dir + '{}_ins_label.npy'.format(fn))
                nl = np.load(self.npy_dir + '{}_nl.npy'.format(fn))
                pass
            file_name.append(self.train_file_list[idx])
            xyz = xyz - xyz.min(0)
            xyz, rgb, nl = self.dataAugment(xyz, rgb, nl, i, jitter=True, flip=True, rot=True, scale=True, elastic=True)

            # #####mix up
            if self.mixup == True:
                mix_id = np.floor(np.random.rand() * len(self.train_file_list)).astype(np.int64)
                mix_fn = self.train_file_list[mix_id]
                mix_xyz = SA.attach("shm://{}_xyz".format(mix_fn)).copy()
                mix_rgb = SA.attach("shm://{}_rgb".format(mix_fn)).copy()
                mix_sem = SA.attach("shm://{}_sem_label".format(mix_fn)).copy()
                mix_ins = SA.attach("shm://{}_ins_label".format(mix_fn)).copy()
                mix_nl = SA.attach("shm://{}_nl".format(mix_fn)).copy()
                mix_xyz, mix_rgb, mix_nl = self.dataAugment(mix_xyz, mix_rgb, mix_nl, i, jitter=True, flip=True, rot=True,
                                                            scale=True, elastic=True)
                # #merge scene
                xyz = np.concatenate((xyz, mix_xyz), axis=0)
                rgb = np.concatenate((rgb, mix_rgb), axis=0)
                sem = np.concatenate((sem, mix_sem), axis=0)
                nl = np.concatenate((nl, mix_nl), axis=0)
                ins_num_a = ins.max() + 1
                mix_ins[np.where(mix_ins != -100)] += ins_num_a
                ins = np.concatenate((ins, mix_ins), axis=0)

            # ########crop scenes
            max_tries = 5
            while (max_tries > 0):
                xyz_crop, valid_ind = self.crop(xyz)
                if valid_ind.sum() >= self.min_crop_p:
                    xyz = xyz_crop
                    break
                max_tries = max_tries -1
            xyz = xyz - xyz.min(0)
            xyz = xyz[valid_ind, ...]
            rgb = rgb[valid_ind, ...]
            sem = sem[valid_ind, ...]
            nl = nl[valid_ind, ...]
            ins = self.getCroppedInstLabel(ins, valid_ind)

            # ------------------------------- Voxel and Batch -------------------------
            feats_rgb_normal_line = np.concatenate((rgb, nl), axis=1).astype(np.float32)
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(xyz, feats_rgb_normal_line, \
                                                                                         quantization_size=self.voxel_size,
                                                                                         return_index=True,
                                                                                         return_inverse=True)
            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]

            # ## get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz, ins.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            ins[np.where(ins != -100)] += total_inst_num
            total_inst_num += inst_num

            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))

            sem_batch.append(torch.from_numpy(sem))
            v2p_index_batch.append(v2p_index)
            ins_batch.append(torch.from_numpy(ins.astype(np.float32)))
            inst_info_batch.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            pass
            #  merge all the scenes in the batch
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        sem_batch = torch.cat(sem_batch, 0).to(torch.int64)
        ins_batch = torch.cat(ins_batch, 0).to(torch.int64)
        inst_info_batch = torch.cat(inst_info_batch, 0).to(torch.float32)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)

        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'sem': sem_batch, 'ins': ins_batch, 'inst_info': inst_info_batch, 'instance_pointnum': instance_pointnum,
                'v2p_index': v2p_index_batch, 'fn': file_name}

    def valMerge(self, id):
        xyz_voxel = []
        feat_voxel = []

        xyz_original = []

        sem_batch = []
        ins_batch = []
        inst_info_batch = []
        v2p_index_batch = []
        instance_pointnum = []  # (total_nInst), int

        total_inst_num = 0
        total_voxel_num = 0

        file_name = []
        id = id + id + id
        for i, idx in enumerate(id):
            fn = self.val_file_list[idx]  # get shm name
            if self.cache:
                xyz = SA.attach("shm://{}_xyz".format(fn)).copy()
                rgb = SA.attach("shm://{}_rgb".format(fn)).copy()
                sem = SA.attach("shm://{}_sem_label".format(fn)).copy()
                ins = SA.attach("shm://{}_ins_label".format(fn)).copy()
                nl = SA.attach("shm://{}_nl".format(fn)).copy()
                sup = SA.attach("shm://{}_sup".format(fn)).copy()
            else:
                xyz = np.load(self.npy_dir + '{}_xyz.npy'.format(fn))
                rgb = np.load(self.npy_dir + '{}_rgb.npy'.format(fn))
                sem = np.load(self.npy_dir + '{}_sem_label.npy'.format(fn))
                ins = np.load(self.npy_dir + '{}_ins_label.npy'.format(fn))
                nl = np.load(self.npy_dir + '{}_nl.npy'.format(fn))
                sup = np.load(self.npy_dir + '{}_sup.npy'.format(fn))
                pass
            file_name.append(self.val_file_list[idx])

            xyz, rgb, nl = self.dataAugment(xyz, rgb, nl, i, jitter=False, flip=False, rot=False, scale=False, elastic=False)
            ins = self.getInstLabel(ins)
            # ------------------------------- Voxel and Batch -------------------------
            feats_rgb_normal_line = np.concatenate((rgb, nl), axis=1).astype(np.float32)
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(xyz, feats_rgb_normal_line, \
                                                                                         quantization_size=self.voxel_size,
                                                                                         return_index=True,
                                                                                         return_inverse=True)
            v2p_index = inverse_index + total_voxel_num
            total_voxel_num = total_voxel_num + index.shape[0]

            # ## get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz, ins.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            ins[np.where(ins != -100)] += total_inst_num
            total_inst_num += inst_num

            #  merge the scene to the batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_original.append(torch.from_numpy(xyz))

            sem_batch.append(torch.from_numpy(sem))
            v2p_index_batch.append(v2p_index)
            ins_batch.append(torch.from_numpy(ins.astype(np.float32)))
            inst_info_batch.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            pass
            #  merge all the scenes in the batch
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original, 0).to(torch.float32)
        sem_batch = torch.cat(sem_batch, 0).to(torch.int64)
        ins_batch = torch.cat(ins_batch, 0).to(torch.int64)
        inst_info_batch = torch.cat(inst_info_batch, 0).to(torch.float32)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)

        return {'xyz_voxel': xyz_voxel_batch, 'feat_voxel': feat_voxel_batch, 'xyz_original': xyz_original,
                'sem': sem_batch, 'ins': ins_batch, 'inst_info': inst_info_batch, 'instance_pointnum': instance_pointnum,
                'v2p_index': v2p_index_batch, 'fn': file_name, 'sup': sup}


