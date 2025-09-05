from networks.Unet import UNet
from networks.unet_mso import CACNet2d
from networks.MCNet2d import MCNet2d_v1
def create_model(name='unet',in_chns=1, class_num=2, has_dropout=True, has_proj=False):
    # Network definition
    if name == 'unet':
        net = UNet(in_channels=in_chns, out_channels=class_num)

    elif name == 'CAC':
        "emb_num是倒数第二层的特征图通道数"
        net = CACNet2d(in_chns=in_chns, class_num=class_num, emb_num=16)

    elif name == 'MCNet_v1':
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num)


    model = net.cuda()
    return model
