# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/6/28  下午10:41
# File Name: decode_scannet.py
# IDE: PyCharm

import os
import json
import csv
import glob
import datetime
import torch
import numpy as np
import SharedArray as SA
from plyfile import PlyData

import segmentator
from lib.PB_lib.torch_io.pbnet_ops import get_normal_line

# ##if use cuda to decode the normal
use_cuda_flag = True

# ###define the file path
SCANNET_DIR = './datasets/scannetv2/'
LABEL_MAP_FILE = './datasets/scannetv2/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
OUTPUT_FOLDER = './datasets/scannetv2/npy/'

##################################################decode data###########################################
def represents_int(s):
    #  if string s represents an int
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


# ### read XYZ RGB for each vertex. ( RGB values are in 0-255)
def read_mesh_vertices_rgb(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']

        faces = plydata['face'].data['vertex_indices']
    return vertices, faces


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def surface_normal_area(face, vertex):
    normals = list()
    areas = list()
    vertex_to_face = [[] for i in range(len(vertex))]
    for fid, f in enumerate(face):
        # f = f[0]
        va, vb, vc = f[0], f[1], f[2]
        vertex_to_face[va].append(fid)
        vertex_to_face[vb].append(fid)
        vertex_to_face[vc].append(fid)

        a = vertex[vb] - vertex[va]
        b = vertex[vc] - vertex[va]
        normal = np.cross(a, b)
        area = np.dot(normal, normal) / 2.0
        normalized_normal = normal / np.linalg.norm(normal)
        normals.append(normalized_normal)
        areas.append(area)
    return np.array(normals), np.array(areas), vertex_to_face


# ###use CPU calculate normal (too slow)
def vertex_normal(vertex_to_face, normal, areas):
    vertex_normals = list()
    num_vertex = len(vertex_to_face)
    for vid in range(num_vertex):
        adj_faces = vertex_to_face[vid]
        if len(adj_faces) == 0:  # single point with no adjancy points
            vertex_normals.append([0, 0, 1])
            continue
        adj_faces_area = np.expand_dims(np.array(areas[adj_faces]), axis=-1)
        adj_faces_normal = np.array(normal[adj_faces])
        avg_normal = (adj_faces_normal * adj_faces_area) / np.sum(adj_faces_area)
        avg_normal = np.sum(avg_normal, axis=0)
        normalized_normal = avg_normal / np.linalg.norm(avg_normal)
        # if np.isclose(np.linalg.norm(avg_normal), 0.0):
        #    print('-------------------')
        #    print(len(adj_faces))
        #    print('-------------------')
        #    print('-------------------')
        #    print(adj_faces_area.shape, adj_faces_normal.shape, adj_faces_area, adj_faces_normal)
        #    print(adj_faces_normal * adj_faces_area)
        #    print(np.sum(adj_faces_area))
        #    print((adj_faces_normal * adj_faces_area) / np.sum(adj_faces_area))
        #    print(avg_normal, np.linalg.norm(avg_normal), adj_faces_area, adj_faces_normal)
        #    print('-------------------')
        vertex_normals.append(normalized_normal)
    return np.array(vertex_normals)


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file):
    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    mesh_vertices, faces = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    if use_cuda_flag == False:
        normal, areas, vertex_to_face = surface_normal_area(faces, mesh_vertices[:, :3])
        normal_line_vertex = vertex_normal(vertex_to_face, normal, areas)
    if use_cuda_flag == True:
        normal_line_vertex = get_normal_line(mesh_vertices[:, :3], face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(mesh_vertices.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()


    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                         xmax - xmin, ymax - ymin, zmax - zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox

    return mesh_vertices, label_ids, instance_ids, instance_bboxes, object_id_to_label_id, face_npy, normal_line_vertex, superpoint


def export_one_scan(scan_name, output_filename_prefix, data_split):
    mesh_file = os.path.join(SCANNET_DIR, data_split, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, data_split, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, data_split, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, data_split, scan_name + '.txt') # includes axisAlignment info for the train set scans.
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic,  face_npy, normal_line_vertex, superpoint = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask, :]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]
    normal_line_vertex = normal_line_vertex[mask, :]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    remapper = np.ones(150) * (-100)
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        remapper[x] = i
    semantic_labels = remapper[semantic_labels]
    # ####split xyz & rgb
    xyz = mesh_vertices[:, :3]
    rgb = mesh_vertices[:, 3:]

    np.save(output_filename_prefix+'_xyz.npy', xyz)
    np.save(output_filename_prefix + '_rgb.npy', rgb)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)
    np.save(output_filename_prefix + '_nl.npy', normal_line_vertex)
    np.save(output_filename_prefix + '_face.npy', face_npy)
    np.save(output_filename_prefix + '_sup.npy', superpoint)


def export_one_scan_test(scan_name, output_filename_prefix, data_split):
    mesh_file = os.path.join(SCANNET_DIR, data_split, scan_name + '_vh_clean_2.ply')
    mesh_vertices, faces = read_mesh_vertices_rgb(mesh_file)

    ##get normal line
    face_npy = faces.tolist()
    face_npy = np.concatenate(face_npy).reshape(-1, 3)
    #  # normal line
    if use_cuda_flag == False:
        normal, areas, vertex_to_face = surface_normal_area(faces, mesh_vertices[:, :3])
        normal_line_vertex = vertex_normal(vertex_to_face, normal, areas)
    if use_cuda_flag == True:
        normal_line_vertex = get_normal_line(mesh_vertices[:, :3], face_npy)

    # ###get superpoints
    superpoint = segmentator.segment_mesh(torch.from_numpy(mesh_vertices.astype(np.float32)),
                                          torch.from_numpy(face_npy.astype(np.int64))).numpy()

    # ####split xyz & rgb
    xyz = mesh_vertices[:, :3]
    rgb = mesh_vertices[:, 3:]

    np.save(output_filename_prefix + '_xyz.npy', xyz)
    np.save(output_filename_prefix + '_rgb.npy', rgb)
    np.save(output_filename_prefix + '_face.npy', face_npy)
    np.save(output_filename_prefix + '_nl.npy', normal_line_vertex)
    np.save(output_filename_prefix + '_sup.npy', superpoint)
    pass


def batch_export(data_split):
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))
        os.mkdir(OUTPUT_FOLDER)
    Split_name = glob.glob(SCANNET_DIR + data_split + "/*clean_2.ply")
    Split_name.sort()
    for scan_name in Split_name:
        scan_name = scan_name.split('/')[-1]
        scan_name = scan_name[:12]
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        if os.path.isfile(output_filename_prefix + '_sup.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, data_split)
        except:
            print('Failed export scan: %s' % (scan_name))
        print('-' * 20 + 'done')

def batch_export_test():
    Split_name = glob.glob(SCANNET_DIR + "test/*clean_2.ply")
    Split_name.sort()
    for i, scan_name in enumerate(Split_name):
        scan_name = scan_name.split('/')[-1]
        scan_name = scan_name[:12]
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        if os.path.isfile(output_filename_prefix + '_sup.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done {}'.format(i))
            continue
        try:
            export_one_scan_test(scan_name, output_filename_prefix, 'test')
        except:
            print('Failed export scan: %s' % (scan_name))
        print('-' * 20 + 'done')

    pass

##################################################create shm###########################################

# #####shm create
def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def create_shm_train(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/{}_nl".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            xyz = np.load(npy_path + '{}_xyz.npy'.format(fn))
            rgb = np.load(npy_path + '{}_rgb.npy'.format(fn))
            sem_label = np.load(npy_path + '{}_sem_label.npy'.format(fn))
            ins_label = np.load(npy_path + '{}_ins_label.npy'.format(fn))
            nl = np.load(npy_path + '{}_nl.npy'.format(fn))
            # ####write share memory
            sa_create("shm://{}_xyz".format(fn), xyz)
            sa_create("shm://{}_rgb".format(fn), rgb)
            sa_create("shm://{}_sem_label".format(fn), sem_label)
            sa_create("shm://{}_ins_label".format(fn), ins_label)
            sa_create("shm://{}_nl".format(fn), nl)


def create_shm_val(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/{}_nl".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            xyz = np.load(npy_path + '{}_xyz.npy'.format(fn))
            rgb = np.load(npy_path + '{}_rgb.npy'.format(fn))
            sem_label = np.load(npy_path + '{}_sem_label.npy'.format(fn))
            ins_label = np.load(npy_path + '{}_ins_label.npy'.format(fn))
            sup = np.load(npy_path + '{}_sup.npy'.format(fn))
            nl = np.load(npy_path + '{}_nl.npy'.format(fn))
            # ####write share memory
            sa_create("shm://{}_xyz".format(fn), xyz)
            sa_create("shm://{}_rgb".format(fn), rgb)
            sa_create("shm://{}_sem_label".format(fn), sem_label)
            sa_create("shm://{}_ins_label".format(fn), ins_label)
            sa_create("shm://{}_sup".format(fn), sup)
            sa_create("shm://{}_nl".format(fn), nl)


def create_shm_test(List_name, npy_path):
    for i, list_name in enumerate(List_name):
        fn = list_name  # get shm name
        if not os.path.exists("/dev/shm/{}_nl".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            # ####read npy file
            xyz = np.load(npy_path + '{}_xyz.npy'.format(fn))
            rgb = np.load(npy_path + '{}_rgb.npy'.format(fn))
            sup = np.load(npy_path + '{}_sup.npy'.format(fn))
            nl = np.load(npy_path + '{}_nl.npy'.format(fn))
            # ####write share memory
            sa_create("shm://{}_xyz".format(fn), xyz)
            sa_create("shm://{}_rgb".format(fn), rgb)
            sa_create("shm://{}_sup".format(fn), sup)
            sa_create("shm://{}_nl".format(fn), nl)


if __name__=='__main__':
    batch_export('train')
    batch_export('val')
    batch_export_test()
    train_list = np.loadtxt(SCANNET_DIR + 'scannetv2_train.txt', dtype='str')
    val_list = np.loadtxt(SCANNET_DIR + 'scannetv2_val.txt', dtype='str')
    test_list = np.loadtxt(SCANNET_DIR + 'scannetv2_test.txt', dtype='str')
    create_shm_train(train_list, OUTPUT_FOLDER)
    create_shm_val(val_list, OUTPUT_FOLDER)
    create_shm_test(test_list, OUTPUT_FOLDER)

