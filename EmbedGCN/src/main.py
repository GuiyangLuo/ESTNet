import numpy as np
import os

import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import yaml
from types import SimpleNamespace as SN
import torch
import  utils.util as util
import numpy as np
import argparse
import time
import random
import matplotlib.pyplot as plt
import pickle
from runners.train import  train
from runners.test import  test
from vis.hyper_parameters import  main_plot

def _get_config(params , arg_name='--alg', subfolder='algs'):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    if config_name is not None:

        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f,Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def _get_config_dataset(yamlname = 'dataset'):


    with open(os.path.join(os.path.dirname(__file__), "config", "algs","{}.yaml".format(yamlname)),
              "r") as f:
        try:
            dataset_config_dict = yaml.load(f,Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(yamlname, exc)
    return dataset_config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def save_results(results,args):
    with open("dict_data_multiply_4.pkl", 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(results, fo)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getFileSize(dir_for_logs, filePath, size=0):
    filePath = os.path.join(dir_for_logs, filePath)
    for root, dirs, files in os.walk(filePath):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return size/1024/1204
def train_main():
    params = deepcopy(sys.argv)
    params.append('--alg=embedgcn')
    alg_config = _get_config(params, "--alg", "algs")
    dataset_config = _get_config_dataset(alg_config['dataset_name'])
    config = recursive_dict_update(alg_config, dataset_config)
    args = SN(**config)

    # time.sleep(int(60*60*4))
    # args.random_seed = int(np.random.rand() * 10000)
    if args.generate_dataset:
        print('Begin to generate data with the above configurations.')

    # for layers in [[1,2,2,3],]:
    # for layers in [[1,1,2,2]]:
    for layers in [[1, 2, 3, 4], ]:
        args.resnet_layers = layers
        cur_time = time.strftime("%d%H%M%S", time.localtime())

        name = f"{cur_time}_EA_rand{args.random_seed}_in{args.channels_conv_start_layer}_layers{'.'.join([str(x) for x in args.resnet_layers])}_out{args.channels_conv_extractor_out}_neighbors{args.top_k_neighbors}_gcn{args.number_of_scales}_{args.fusion}_{args.dataset_name}"
        # name = "EA_rand3998_in32_layers1.2.3.4_out156_neighbors16_gcn3_concat_pems-bay"
        args.model_save_dir = str(args.model_save_dir).format(name)
        args.tensorboard_dir = str(args.tensorboard_dir).format(name)
        print('All the configurations:', args)
        setup_seed(args.random_seed)
        train(args)


if __name__ == '__main__':
    train_main()



