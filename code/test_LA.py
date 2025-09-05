import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import create_model
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='model/RAML_16_labeled/', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='Mutual', help='model_name')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')


def Inference(FLAGS):
    num_classes = 2
    #图片保存路径
    test_save_path = "./Prediction/{}".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    #模型保存路径
    save_mode_path = "./{}".format(FLAGS.exp)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric, std = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric, std


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    avg_metric, std = Inference(FLAGS)
    print('avg_metric for dice, jc, hd, asd:', avg_metric)
    print('std for dice, jc, hd, asd:', std)
