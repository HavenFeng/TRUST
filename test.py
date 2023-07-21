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

import argparse
import ast

from lib.model import TRUST
from lib import util

### default dict
fixed_dict = {
            # FLAME
            'topology_path': './data/head_template.obj',
            'flame_model_path': './data/generic_model.pkl',
            'flame_lmk_embedding_path': './data/landmark_embedding.npy',

            'BFM_tex_path': './data/FLAME_albedo_from_BFM.npz',
            'BalanceAlb_tex_path': './data/BalanceAlb_model.npz',
            'lightprobe_normal_path': './data/lightprobe_normal_images.npy',
            'lightprobe_albedo_path': './data/lightprobe_albedo_images.npy',

            'n_shape': 100,
            'n_tex': 54,
            'n_exp': 50,
            'n_pose': 6,
            'n_cam': 3,
            'n_light': 27,
            'n_scenelight': 3,
            'n_facelight': 27,

            'image_size': 224,
            'scene_size': 224,
            'uv_size': 256,
            'scale_min': 1.4,
            'scale_max': 1.8,
            'trans_scale': 0,
            'use_flip': False,
            'num_worker': 1,
            'n_train': 1000,


        }


config_test = {
    'test_data': 'benchmark_val',
    'dataname': 'benchmark_val',
    'batch_size': 16,
    'euler': False,

    'tex_type': 'BalanceAlb', # BalanceAlb is the albedo model for the paper's results
    'pretrain_scene': False,
    'pretrain_albedo': False,
    'scene_fix': False,
    'no_cond': False,
    'pho_type': 'tex',
    'resume_training': True,
    'pretrained_modelpath_flame': './data/deca_model.tar',
    'pretrained_modelpath_status': '',
    'savefolder': './test_outputs',

}

def merge_config(config, config_current):
    config_dict = config_current
    for key in config_dict:
        setattr(config, key, config_dict[key])

    if config.resume_training and os.path.exists(config.pretrained_modelpath_flame) is not True:
        print('check pretrained model path')
    return config


def save_config(config, config_path):
    config_dict = config.__dict__
    import yaml
    with open(config_path, 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing TRUST regressors')

    parser.add_argument('--test_folder', default='', type=str,
                        help='test folder path')
    parser.add_argument('--test_split', default='val', type=str,
                        help='test data path')
    parser.add_argument('--test_data', default='', type=str,
                        help='benchmark_val and pexels_test')

    args = parser.parse_args()

    default_config = util.dict2obj(fixed_dict)

    ### test
    config = merge_config(default_config, config_test)

    if args.test_folder !='':
        config.n_scenelight = 3

        config.pretrained_modelpath_scene ='{}/E_scene_light_{}.tar'.format(args.test_folder, config.tex_type)
        config.pretrained_modelpath_facel = '{}/E_face_light_{}.tar'.format(args.test_folder, config.tex_type)
        config.pretrained_modelpath_albedo = '{}/E_albedo_{}.tar'.format(args.test_folder, config.tex_type)

        if args.test_data!='':
            config.test_data = args.test_data
            config.dataname = args.test_data

        if args.test_split!='':
            config.test_split = args.test_split
            config.dataname = 'benchmark_split_{}'.format(args.test_split)

        trust = TRUST(config)

        util.check_mkdir(config.savefolder + '/' + config.dataname)

        vis_path = os.path.sep.join([config.savefolder, config.dataname, 'test_images_vis'])
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        trust.test(return_params=True)
