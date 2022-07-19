# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de


import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from scipy.ndimage import morphology
from skimage.io import imsave
import cv2

import pyshtools as pysh


# copy from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    
def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def rotateSH(sh_input, rot_angle, rot_axis):
    ls = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
    ms = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2])
    mneg_mask = (ms < 0).astype(np.int)

    # (csphase=1 => exclude)
    __OUR_SH_CSPHASE = 1
    __OUR_SH_NORM = 'ortho'

    def _to_pysh(oldsh):
        # old: [l0,m0; l1,m-1; l1,m0; l1,m1; l2,m-2, l2,m-1, l2,m0, l2,m1, l2,m2]
        # newsh = np.zeros(2, 3, 3)

        newsh = []
        for c in range(3):
            curr_sh = pysh.SHCoeffs.from_zeros(lmax=2, normalization=__OUR_SH_NORM, csphase=__OUR_SH_CSPHASE)
            values = oldsh[:, c]
            # print("&&&")
            # print(values)
            curr_sh.set_coeffs(values, ls, ms)
            # print(curr_sh.coeffs)
            # print("&&&")
            # L0,m0
            # curr_sh.set_coeffs( oldsh[0, c],  )

            newsh.append(curr_sh)

        return newsh

    def _from_pysh(shlist):

        out = []
        for c in range(3):
            # convert back to orthonormalized
            # _sh = shlist[c].convert('ortho')
            _sh_orth = shlist[c].to_array(normalization=__OUR_SH_NORM, csphase=__OUR_SH_CSPHASE)
            # print("///")
            # print(_sh_orth)
            _sh_1D = _sh_orth[mneg_mask, ls, np.abs(ms)].ravel()
            # print(_sh_1D)
            out.append(_sh_1D[:, None])

        out = np.concatenate(out, axis=1)
        return out

    # convert to pysh format
    _sh = _to_pysh(sh_input)

    # rotate each channel
    rotM = pysh.rotate.djpi2(2)
    rot = [0, 0, 0]
    rot[rot_axis] = rot_angle * np.pi / 180.
    rot_sh = []

    for c in range(3):
        _sh_for_rot = _sh[c].convert('4pi', csphase=1)  # needed for rotation
        _sh_rot = pysh.rotate.SHRotateRealCoef(_sh_for_rot.coeffs, rot, rotM)
        _sh_rot = pysh.SHCoeffs.from_array(_sh_rot, normalization='4pi', csphase=1)
        rot_sh.append(_sh_rot)

    rot_sh = _from_pysh(rot_sh)
    return torch.as_tensor(rot_sh)


# --------------------------- euler angle to rotatio vector
def euler2quat_conversion_sanity_batch(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    # quaternion = torch.zeros([batch_size, 4])
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion

def quaternion_to_angle_axis(quaternion: torch.Tensor):
    """Convert quaternion vector to angle axis of rotation. TODO: CORRECT

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta).to(quaternion.device)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion).to(quaternion.device)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def euler2aa_batch(r):
    return quaternion_to_angle_axis(euler2quat_conversion_sanity_batch(r))

def batch_rodrigues(theta):
    # theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def aa2euler_batch(r):
    return rot_mat_to_euler(batch_rodrigues(r))


def deg2rad(tensor):
    """Function that converts angles from degrees to radians.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    return tensor * torch.tensor(math.pi).to(tensor.device).type(tensor.dtype) / 180.

def batch_orth_proj(X, camera):
    '''
        X is N x num_pquaternion_to_angle_axisoints x 3
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    Xn = (camera[:, :, 0:1] * X_trans)
    # import ipdb; ipdb.set_trace()
    return Xn

# -------------------------------------- image processing


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [batch_size, 3] tensor containing X, Y, and Z angles.
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [batch_size, 3, 3]. rotation matrices.
    '''
    angles = angles*(np.pi)/180.
    s = torch.sin(angles)
    c = torch.cos(angles)

    cx, cy, cz = (c[:, 0], c[:, 1], c[:, 2])
    sx, sy, sz = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0]).to(angles.device)
    ones = torch.ones_like(s[:, 0]).to(angles.device)

    # Rz.dot(Ry.dot(Rx))
    R_flattened = torch.stack(
    [
      cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
          -sy,                cy * sx,                cy * cx,
    ],
    dim=0) #[batch_size, 9]
    R = torch.reshape(R_flattened, (-1, 3, 3)) #[batch_size, 3, 3]
    return R

def binary_erosion(tensor, kernel_size=5):
    # tensor: [bz, 1, h, w]. 
    device = tensor.device
    mask = tensor.cpu().numpy()
    structure=np.ones((kernel_size,kernel_size))
    new_mask = mask.copy()
    for i in range(mask.shape[0]):
        new_mask[i,0] = morphology.binary_erosion(mask[i,0], structure)
    return torch.from_numpy(new_mask.astype(np.float32)).to(device)

def flip_image(src_image, kps):
    '''
        purpose:
            flip a image given by src_image and the 2d keypoints
        flip_mode: 
            0: horizontal flip
            >0: vertical flip
            <0: horizontal & vertical flip
    '''
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        kps[:, :] = kps[kp_map]

    return src_image, kps

# -------------------------------------- io
def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                # print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            # print('copy param {} failed'.format(k))
            continue

def check_mkdir(path):
    if not os.path.exists(path):
        print('creating %s' % path)
        os.makedirs(path)

def check_mkdirlist(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            print('creating %s' % path)
            os.makedirs(path)

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()


def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

# original saved file with DataParallel
def remove_module(state_dict):
# create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def dict_tensor2npy(tensor_dict):
    npy_dict = {}
    for key in tensor_dict:
        npy_dict[key] = tensor_dict[key][0].cpu().numpy()
    return npy_dict
        
# ---------------------------------- visualization
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image,(int(st[0]), int(st[1])), 1, c, 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), 1)

    return image

def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    import ipdb;ipdb.set_trace()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)

    return image

def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color = 'g', isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):

        image = images[i]
        image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
        if isScale:
            predicted_landmark = predicted_landmarks[i]*image.shape[0]/2 + image.shape[0]/2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(vis_landmarks[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
    return vis_landmarks


####################
def calc_aabb(ptSets):
    if not ptSets or len(ptSets) == 0:
        return False, False, False

    ptLeftTop = np.array([ptSets[0][0], ptSets[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for pt in ptSets:
        ptLeftTop[0] = min(ptLeftTop[0], pt[0])
        ptLeftTop[1] = min(ptLeftTop[1], pt[1])
        ptRightBottom[0] = max(ptRightBottom[0], pt[0])
        ptRightBottom[1] = max(ptRightBottom[1], pt[1])

    return ptLeftTop, ptRightBottom, len(ptSets) >= 5


def cut_image(filePath, kps, expand_ratio, leftTop, rightBottom):
    originImage = cv2.imread(filePath)
    height = originImage.shape[0]
    width = originImage.shape[1]
    channels = originImage.shape[2] if len(originImage.shape) >= 3 else 1

    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)

    # remove extra space.
    # leftTop, rightBottom = shrink(leftTop, rightBottom, width, height)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop = [int(leftTop[0]), int(leftTop[1])]
    rightBottom = [int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)]

    dstImage = np.zeros(shape=[rightBottom[1] - leftTop[1], rightBottom[0] - leftTop[0], channels], dtype=np.uint8)
    dstImage[:, :, :] = 0

    offset = [lt[0] - leftTop[0], lt[1] - leftTop[1]]
    size = [rb[0] - lt[0], rb[1] - lt[1]]

    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0], :]
    return dstImage, off_set_pts(kps, leftTop)

def flip_image(src_image, kps):
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        kps[:, :] = kps[kp_map]

    return src_image, kps


def draw_lsp_14kp__bone(src_image, pts):
    bones = [
        [0, 1, 255, 0, 0],
        [1, 2, 255, 0, 0],
        [2, 12, 255, 0, 0],
        [3, 12, 0, 0, 255],
        [3, 4, 0, 0, 255],
        [4, 5, 0, 0, 255],
        [12, 9, 0, 0, 255],
        [9, 10, 0, 0, 255],
        [10, 11, 0, 0, 255],
        [12, 8, 255, 0, 0],
        [8, 7, 255, 0, 0],
        [7, 6, 255, 0, 0],
        [12, 13, 0, 255, 0]
    ]

    for pt in pts:
        if pt[2] > 0.2:
            cv2.circle(src_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        if pa[2] > 0.2 and pb[2] > 0.2:
            cv2.line(src_image, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 2)


def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))

    if normalize:
        src_image = (src_image.astype(np.float) / 255) * 2.0 - 1.0

    return src_image
