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

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2


from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import PIL.Image

'''
images and keypoints: nomalized to [-1,1]
'''
########################## testing

class PexelsDataset(Dataset):
    def __init__(self, image_size, scene_size, scale, trans_scale=0):
        self.image_size = image_size
        self.scene_folder = './pexels-diverse/dataset/images'
        self.image_folder = './pexels-diverse/dataset/crops'
        self.kpt_folder = './pexels-diverse/dataset/lmks'
        datafile = './pexels-diverse/dataset/img-list__crop.txt'

        self.image_size = image_size
        self.scene_size = scene_size

        self.data_lines = open(datafile).read().splitlines()

        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        # image_path = os.path.sep.join([self.image_folder, self.data_lines[idx]])
        # scene_name = '/'.join(image_path.split('/')[-3:-1])
        # # image_path = self.data_lines[idx]
        # # scene_name = image_path.split('/')[-2]
        #
        # image_name = image_path.split('/')[-1][:-4]
        # scene_path = os.path.sep.join([self.scene_folder, scene_name + '.jpg'])
        # kpt_path = os.path.sep.join([self.kpt_folder, scene_name, image_name + '.npy'])

        image_path = self.data_lines[idx]
        scene_name = image_path.split('/')[-3]
        image_name = '/'.join(image_path.split('/')[-2:])[:-4]
        scene_path = image_path
        kpt_path = image_path[:-4] + '.npy'

        image = imread(image_path) / 255.
        kpt = np.load(kpt_path)[:, :2]
        kpt = (kpt + 1) / 2 * self.image_size # only for pexels

        scene_image = imread(scene_path) / 255.

        ### crop information
        tform = self.crop(image, kpt)
        ## crop
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

        # random crop scene
        square_scene = self.scene_crop(scene_image)

        ###
        images_array = torch.from_numpy(cropped_image.transpose(2, 0, 1)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)  # K,224,224,3

        scene_array = torch.from_numpy(square_scene.transpose(2, 0, 1)).type(dtype=torch.float32)


        data_dict = {
            'face_image': images_array,
            'face_landmark': kpt_array,
            'scene_image': scene_array,
            'face_image_name': image_name,
            'scene_name': scene_name,
        }

        return data_dict

    def crop(self, image, kpt):
        h_sort = np.sort(kpt[:, 1])
        w_sort = np.sort(kpt[:, 0])
        lmk_h_min = h_sort[1]
        lmk_h_max = h_sort[-2]
        lmk_w_min = w_sort[1]
        lmk_w_max = w_sort[-2]

        left, top, right, bottom = lmk_w_min, lmk_h_min, lmk_w_max, lmk_h_max

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def scene_crop(self, image):
        scene_h, scene_w = image.shape[:2]
        if scene_w > scene_h:
            sq_size = scene_h
            random_left = np.random.randint(scene_w - sq_size)
            square_scene = image[0:sq_size, random_left:random_left + sq_size]
            square_scene = cv2.resize(square_scene, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        elif scene_h > scene_w:
            sq_size = scene_w
            random_top = np.random.randint(scene_h - sq_size)
            square_scene = image[random_top: random_top+sq_size, 0:sq_size]
            square_scene = cv2.resize(square_scene, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        else:
            square_scene = cv2.resize(image, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        return square_scene


class AgoraFaceDataset_test(Dataset):
    def __init__(self, image_size, scene_size, scale, trans_scale=0, split='val'):
        self.image_size = image_size

        if split=='val':
            self.scene_folder = './FAIR_benchmark/validation_set/full_image'
            self.kpt_folder = './FAIR_benchmark/validation_set/crop-lmks'
            datafile = './FAIR_benchmark/validation_set/validation_crop_files.txt'

        elif split=='test':
            self.scene_folder = './FAIR_benchmark/test_set/full_image'
            self.kpt_folder = './FAIR_benchmark/test_set/crop-lmks'
            datafile = './FAIR_benchmark/test_set/test_crop_files.txt'

        else:
            print('please check split input')
            exit()

        self.image_size = image_size
        self.scene_size = scene_size

        self.data_lines = open(datafile).read().splitlines()

        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        image_path = self.data_lines[idx]
        scene_name = image_path.split('/')[-2]
        image_name = image_path.split('/')[-1][:-4]
        scene_path = os.path.sep.join([self.scene_folder, scene_name + '.png'])
        kpt_path = os.path.sep.join([self.kpt_folder, scene_name, image_name + '.npy'])

        image = imread(image_path) / 255.
        kpt = np.load(kpt_path)[:, :2]

        scene_image = PIL.Image.open(scene_path).convert('RGB')
        scene_image = np.asarray(scene_image) / 255.

        ### crop information
        tform = self.crop(image, kpt)
        ## crop
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

        # random crop scene

        random_left = np.random.randint(174)
        square_scene = scene_image[0:224, random_left:random_left+224]
        # cropped_scene = self.random_cropper(square_scene)
        ###
        images_array = torch.from_numpy(cropped_image.transpose(2, 0, 1)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)  # K,224,224,3

        scene_array = torch.from_numpy(square_scene.transpose(2, 0, 1)).type(dtype=torch.float32)

        data_dict = {
            'face_image': images_array,
            'face_landmark': kpt_array,
            'scene_image': scene_array,
            'face_image_name': image_name,
            'scene_name': scene_name,
        }

        return data_dict

    def crop(self, image, kpt):
        h_sort = np.sort(kpt[:, 1])
        w_sort = np.sort(kpt[:, 0])
        lmk_h_min = h_sort[1]
        lmk_h_max = h_sort[-2]
        lmk_w_min = w_sort[1]
        lmk_w_max = w_sort[-2]

        left, top, right, bottom = lmk_w_min, lmk_h_min, lmk_w_max, lmk_h_max

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = 0.5 * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform


