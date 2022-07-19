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
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pickle
import numpy as np
from skimage.io import imread

import cv2
from .renderer import Renderer
from .nets.enc_net import ResnetEncoder, Resnet50Encoder, Resnet50Encoder_v2
from .nets.FLAME import FLAME, FLAMETex
from . import util
from . import datasets_test

torch.backends.cudnn.benchmark = True


class TRUST(object):
    def __init__(self, config, device='cuda'):
        self.device = device
        self.config = config
        self.n_param_deca = 236
        self.n_scenelight = config.n_scenelight
        self.n_facelight = config.n_facelight
        self._create_model()
        self._setup_renderer()


    def _setup_renderer(self):
        self.render = Renderer(self.config.image_size, obj_filename=self.config.topology_path, uv_size=self.config.uv_size).to(self.device)

        self.lightprobe_normal_images = F.interpolate(torch.from_numpy(np.load(self.config.lightprobe_normal_path)).float(), [self.config.image_size, self.config.image_size]).to(self.device)
        self.lightprobe_albedo_images = F.interpolate(torch.from_numpy(np.load(self.config.lightprobe_albedo_path)).float(), [self.config.image_size, self.config.image_size]).to(self.device)
        
    def _create_model(self):
        # encoding
        self.E_flame = ResnetEncoder(outsize=self.n_param_deca).to(self.device) #out: 2048 shape - use deca
        self.E_albedo = Resnet50Encoder_v2(outsize=self.config.n_tex).to(self.device)
        self.E_scene_light = Resnet50Encoder(outsize=self.n_scenelight).to(self.device)
        self.E_face_light = Resnet50Encoder(outsize=self.n_facelight).to(self.device)

        # decoding
        self.flame = FLAME(self.config).to(self.device) # 3DMM shape layer
        self.flametex = FLAMETex(self.config).to(self.device) # texture layer

        ##
        flame_model_path = self.config.pretrained_modelpath_flame
        scene_model_path = self.config.pretrained_modelpath_scene
        facel_model_path = self.config.pretrained_modelpath_facel
        albedo_model_path = self.config.pretrained_modelpath_albedo

        if self.config.resume_training:
            print('trained model found. load {}'.format(flame_model_path))
            checkpoint = torch.load(flame_model_path)
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            if os.path.exists(scene_model_path):
                print('trained model found. load {}'.format(scene_model_path))
                checkpoint = torch.load(scene_model_path)
                util.copy_state_dict(self.E_scene_light.state_dict(), checkpoint['E_scene_light'])

            if os.path.exists(facel_model_path):
                print('trained model found. load {}'.format(facel_model_path))
                checkpoint = torch.load(facel_model_path)
                util.copy_state_dict(self.E_face_light.state_dict(), checkpoint['E_face_light'])

            if os.path.exists(albedo_model_path):
                print('trained model found. load {}'.format(albedo_model_path))
                checkpoint = torch.load(albedo_model_path)
                util.copy_state_dict(self.E_albedo.state_dict(), checkpoint['E_albedo'])

            self.start_epoch = 0
            self.start_iter = 0

        else:
            print('Start training from scratch')
            self.start_epoch = 0
            self.start_iter = 0

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [self.config.n_shape, 50, self.config.n_exp, self.config.n_pose, self.config.n_cam, self.config.n_light]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start+num_list[i]])
            start = start + num_list[i]
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list

    def SH_normalization(self, sh_param):
        norm_sh_param = F.normalize(sh_param, p=1, dim=1)
        return norm_sh_param

    def lightprobe_shading(self, sh):
        bz = sh.shape[0]
        lightprobe_normal = self.lightprobe_normal_images.expand(bz, -1, -1, -1)
        shading_image = self.render.add_SHlight(lightprobe_normal, sh)
        return shading_image

    def SH_convert(self, sh):
        '''
        rotate SH with pi around x axis
        '''
        gt_sh_inverted = sh.clone()
        gt_sh_inverted[:, 1, :] *= -1
        gt_sh_inverted[:, 2, :] *= -1
        gt_sh_inverted[:, 4, :] *= -1
        gt_sh_inverted[:, 7, :] *= -1

        return gt_sh_inverted

    def fuse_light(self, E_scene_light_pred, E_face_light_pred):
        normalized_sh_params = self.SH_normalization(E_face_light_pred)
        lightcode = E_scene_light_pred.unsqueeze(1).expand(-1, 9, -1) * normalized_sh_params

        return lightcode, E_scene_light_pred, normalized_sh_params

    def encoding(self, images, scene_images):
        '''
        :param images:
        :param scene_images:
        :param face_lighting:
        :param scene_lighting:
        :return:
        '''
        # -- encoder
        # coarse
        B, C, H, W = images.size()

        with torch.no_grad():
            parameters = self.E_flame(images)
        code_list = self.decompose_code(parameters)
        shapecode, _, expcode, posecode, cam, _ = code_list

        # ------ image conditioning, scale-normalized spherical harmonic----------------
        if self.config.scene_fix:
            with torch.no_grad():
                E_scene_light_pred = self.E_scene_light(scene_images)
        else:
            E_scene_light_pred = self.E_scene_light(scene_images)

        E_face_light_pred = self.E_face_light(images).reshape(B, 9, 3)

        if self.n_scenelight != 3:
            print('scenelight dim is wrong, should be 3')
            exit()

        lightcode, scale_factors, normalized_sh_params = self.fuse_light(E_scene_light_pred, E_face_light_pred)

        images = images.contiguous().view(B, C, H, W)
        scale_factors_img = scale_factors.contiguous().view(B, scale_factors.size(1), 1, 1).repeat(1, 1, H, W)

        images_cond = torch.cat((scale_factors_img, images), dim=1)

        texcode = self.E_albedo(images_cond)

        batch_size = images.shape[0]
        # -- decoder
        # FLAME - world space
        verts, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode,
                                                     pose_params=posecode)
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        predicted_landmarks[:, :, 1:] = - predicted_landmarks[:, :, 1:]

        albedo = self.flametex(texcode)

        # ------ rendering
        ops = self.render(verts, trans_verts, albedo, lightcode)
        shape_images = self.render.render_shape(verts, trans_verts)

        return scale_factors, normalized_sh_params, texcode, lightcode, albedo, ops, shape_images

    def visualize(self, visdict, savepath):
        grids = {}
        for key in visdict:
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [self.config.image_size, self.config.image_size])).detach().cpu()
        grid = torch.cat(list(grids.values()), 1)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        cv2.imwrite(savepath, grid_image)


    def test(self, return_params=False):
        if self.config.test_data == 'benchmark_val':
            testdata = datasets_test.AgoraFaceDataset_test(image_size=self.config.image_size, scene_size=self.config.scene_size, scale=[self.config.scale_min, self.config.scale_max], trans_scale=self.config.trans_scale, split=self.config.test_split)
        elif self.config.test_data == 'pexels_test':
            testdata = datasets_test.PexelsDataset(image_size=self.config.image_size, scene_size=self.config.scene_size, scale=[self.config.scale_min, self.config.scale_max], trans_scale=self.config.trans_scale)
        else:
            print('please check test data')
            exit()

        test_loader = DataLoader(testdata, batch_size=self.config.batch_size, shuffle=False,
                                  num_workers=self.config.num_worker,
                                  pin_memory=True,
                                  drop_last=False)

        self.E_flame.eval()
        self.E_scene_light.eval()
        self.E_face_light.eval()
        self.E_albedo.eval()

        for iter, batch_data in enumerate(test_loader):
            images = batch_data['face_image'].cuda();
            images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            images_names = batch_data['face_image_name']
            scene_names = batch_data['scene_name']
            scene_images = batch_data['scene_image'].cuda()
            scene_images = scene_images.view(-1, scene_images.shape[-3], scene_images.shape[-2], scene_images.shape[-1])

            scale_factors, normalized_sh_params, texcode, lightcode, albedo, ops, shape_images = self.encoding(images, scene_images)

            batch_size = images.shape[0]

            # images
            predicted_images_alpha = ops['images'] * ops['alpha_images']
            predicted_albedo_images = ops['albedo_images'] * ops['alpha_images']
            predicted_shading = self.lightprobe_shading(self.SH_convert(lightcode))
            # import ipdb;ipdb.set_trace()

            for col_num in range(self.config.batch_size):
                visind = np.arange(col_num, col_num+1)  # self.config.batch_size )
                if self.config.test_data == 'benchmark_val':
                    visdict = {
                        'scene': scene_images[visind],
                        'inputs': images[visind],

                        'predicted_images': predicted_images_alpha[visind],
                        'albedo_images': predicted_albedo_images[visind],
                        'albedo': albedo[visind],
                        'pred_lightprobe': (predicted_shading * self.lightprobe_albedo_images)[visind],

                    }
                elif self.config.test_data == 'pexels_test':
                    visdict = {
                        'scene': scene_images[visind],
                        'inputs': images[visind],
                        'predicted_images': predicted_images_alpha[visind],
                        'albedo_images': predicted_albedo_images[visind],
                        'albedo': albedo[visind],
                        'pred_lightprobe': (predicted_shading * self.lightprobe_albedo_images)[visind],
                    }

                image_name = images_names[col_num]
                scene_name = scene_names[col_num]
                if self.config.test_data == 'benchmark_val':
                    savepath = '{}/{}/{}/{}_{}.jpg'.format(self.config.savefolder, self.config.dataname, 'test_images_vis', scene_name, image_name)
                elif self.config.test_data == 'pexels_test':
                    savepath = '{}/{}/{}/{}_{}.jpg'.format(self.config.savefolder, self.config.dataname, 'test_images_vis', '_'.join(scene_name.split('/')), image_name)

                else:
                    print('please check test_data')
                    exit()

                self.visualize(visdict, savepath)
                print(savepath)
                visdict_albedo = {
                    'albedo': albedo[visind],
                }
                savepath = os.path.sep.join([self.config.savefolder, self.config.dataname, scene_name, image_name + '.jpg'])
                if not os.path.exists(os.path.dirname(savepath)):
                    os.makedirs(os.path.dirname(savepath))
                self.visualize(visdict_albedo, savepath)

                if return_params:
                    param_dict = {
                        'texcode': texcode[visind].detach().cpu().numpy(),
                        'lightcode': lightcode[visind].detach().cpu().numpy(),
                    }
                    savepath = os.path.sep.join([self.config.savefolder, self.config.dataname, scene_name, image_name + '.npy'])
                    if not os.path.exists(os.path.dirname(savepath)):
                        os.makedirs(os.path.dirname(savepath))
                    np.save(savepath, param_dict)
                    with open(savepath, 'wb') as f:
                        pickle.dump(param_dict, f, protocol=2)

