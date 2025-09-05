from networks.vnet import VNet
from networks.MCNet3d import MCNet3d_v1,MCNet3d_v2
from networks.FEDDNET import FEDDNET
from networks.ResNet34 import Resnet34
from networks.Unet3D import UNet3D
from networks.attention_unet import Attention_UNet
from networks.VoxResNet import VoxResNet


def create_model(name='vnet',in_chns=1, class_num=2, has_dropout=True, has_proj=False):
    # Network definition
    if name == 'vnet':
        net = VNet(n_channels=in_chns, n_classes=class_num,n_filters=16, normalization='batchnorm',has_residual=False, has_dropout=has_dropout,has_proj=has_proj)

    elif name == 'unet3d':
        net = UNet3D(in_channels=in_chns, out_channels=class_num, base_channels=32, has_dropout=has_dropout)

    elif name == 'MCNet_v1':
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm',  has_residual=True, has_dropout=has_dropout)

    elif name == 'MCNet_v2':
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_residual=False,has_dropout=has_dropout)

    elif name == 'FEDDNET':
        net = FEDDNET(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_residual=False, has_dropout=has_dropout,has_proj=has_proj)

    model = net.cuda()
    return model
