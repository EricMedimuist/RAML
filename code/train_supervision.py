import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from networks.net_factory_3d import create_model

from utils import ramps, losses
from utils import contrast_loss
from dataloaders.isles2022_process import *
# from dataloaders.isles2018_process import *

# CL
import cleanlab
#损失函数
kl_distance = nn.KLDivLoss(reduction='none')
ce_loss = nn.CrossEntropyLoss(reduction='none')
mse_loss = nn.MSELoss(reduction='none')
contrast_loss_fn = contrast_loss.Contrast()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='ISLES2022', help='dataset_name') # 使用的数据集
parser.add_argument('--root_path', type=str, default='dataset', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default="vnet_supervision_(1)", help='reserved_model_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_epochs', type=int,  default=500, help='maximum epoch number to train')
parser.add_argument('--warmup_epochs', type=int,  default=50, help='maximum epoch number to warmup')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--total_train_samples', type=int, default=40, help='total samples for training')   # 总训练样本数
parser.add_argument('--base_lr', type=float,  default=0.0001, help='initial learning rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()


train_data_path = args.root_path

snapshot_path = "model/{}/{}_{}_labeled/".format(args.dataset_name, args.exp, args.total_train_samples)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



max_epochs = args.max_epochs
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
warmup_epochs=args.warmup_epochs


num_classes = 2
best_dice = 0


def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


if __name__ == "__main__":

    "## make logger file"
    if os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)
    os.makedirs(snapshot_path)



    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



    logging.info(str(args))

    writer = SummaryWriter(snapshot_path + '/log')


    model_1 = create_model(name=args.model, in_chns=1, class_num=2)




    if args.dataset_name == "ISLES2022":
        db_train = ISLES2022(base_dir=train_data_path,train_samples_num=args.total_train_samples,split='train',transform_1=get_train_transform_1())
        db_val = ISLES2022(base_dir=train_data_path, split='test', transform_1=get_val_transform())

    logging.info("total train samples: {} ;total test samples: {};".format(len(db_train),len(db_val)))



    trainloader = DataLoader(db_train, batch_size=4, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val,  batch_size=1,num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    logging.info(" itertations per epoch of trainloader: {} ;itertations per epoch of valloader: {} ".format(len(trainloader),len(val_loader)))




    # model_1_optimizer = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.01)

    model_1_optimizer =optim.Adam(
    params=model_1.parameters(),  # 要优化的参数
    lr=base_lr,                   # 学习率
    betas=(0.9, 0.999),         # 动量参数
    eps=1e-8,                   # 防止除零错误的常数
    weight_decay=0.01,        # L2 正则化（权重衰减）
    amsgrad=False                # 是否使用 AMSGrad
)

    scheduler = optim.lr_scheduler.CosineAnnealingLR( model_1_optimizer, T_max=max_epochs - warmup_epochs, eta_min=0)





    iter_num = 0
    lr_ = base_lr

    "-------------------------训练部分----------------------------"
    for epoch_num in tqdm(range(1,max_epochs+1), ncols=70):
        print("\n")
        model_1.train()


        supervised_loss_epoch_1=0
        # 注意enumerate返回值有两个,一个是训练批次序号，一个是数据（包含训练数据和标签）
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            # print(' epoch_num:{},i_batch:{}'.format(epoch_num,i_batch))



            volume_batch, volume_label = sampled_batch['image'].as_tensor(), sampled_batch['label'].as_tensor()
            volume_label=torch.squeeze(volume_label,dim=1)
            input_batch,label_batch = volume_batch.cuda(), volume_label.cuda()


            outputs_1 = model_1(input_batch)

            # print("outputs_1:",outputs_1)
            # print("label_batch:",label_batch.shape)


            outputs_1_soft = F.softmax( outputs_1, dim=1)
            # print("outputs_1_soft:", outputs_1_soft)




            ## calculate the supervised loss
            "-------------------------------------supervised-----------------------------------------"
            loss_seg_1 = F.cross_entropy(outputs_1, label_batch.long())
            loss_seg_dice_1 = losses.dice_loss(outputs_1_soft[:, 1, :, :, :], label_batch == 1)

            "total supervised loss for this batch"
            supervised_loss_1 = 0.5 * (loss_seg_1 + loss_seg_dice_1)

            "total supervised loss for this epoch"
            supervised_loss_epoch_1 += supervised_loss_1


            model_1_optimizer.zero_grad()
            supervised_loss_1.backward()
            model_1_optimizer.step()

            # "记录每一批次的监督损失"
            # logging.info('supervised_loss_1: %f ; loss_seg_1: %f ; loss_seg_dice_1: %f ; '
            #              % (supervised_loss_1.item(), loss_seg_1.item(), loss_seg_dice_1.item()))


            time1 = time.time()
            iter_num = iter_num + 1



        if epoch_num < warmup_epochs:
            # 线性增加学习率
            lr_ = base_lr * (epoch_num + 1) / warmup_epochs
            for param_group in model_1_optimizer.param_groups:
                param_group['lr'] = lr_
        else:
            # 当预热结束后，使用余弦调度
            scheduler.step()



        logging.info('------------------------第 %s 轮训练结束-----------------------------  ' % (epoch_num))
        "记录每一轮学习率"
        logging.info('learning rate: %f ; ' % (model_1_optimizer.state_dict()['param_groups'][0]['lr']))
        "记录每一轮的平均监督损失"
        supervised_loss_avg_1 = supervised_loss_epoch_1 / len(trainloader)
        logging.info('supervised_loss_epoch_1: %f ;'% (supervised_loss_avg_1))

        "画每一轮的学习率和监督损失"
        writer.add_scalar('learning_rate', model_1_optimizer.state_dict()['param_groups'][0]['lr'], epoch_num)
        writer.add_scalar('loss/supervised_loss_1', supervised_loss_avg_1.item(), epoch_num)




        "-----------------------------每一轮的验证----------------------------------"

        model_1.eval()
        dice_scores_1 = []

        # prompt_dice_list = []
        for i_batch, sampled_batch in enumerate(val_loader):
            volume_batch_val, volume_label_val = sampled_batch['image'].as_tensor(), sampled_batch['label'].as_tensor()
            volume_label_val = torch.squeeze(volume_label_val, dim=1)
            input_batch, label_batch = volume_batch_val.cuda(), volume_label_val.cuda()

            with torch.no_grad():
                outputs_1 = model_1(input_batch)

                "[4, 2, 64,64,64]"
                outputs_1 = F.softmax(outputs_1, dim=1)

                "[4, 64,64,64]"
                outputs_1 = torch.argmax(outputs_1, dim=1)

                "[4, 64,64,64]"
                label_batch = (label_batch > 0.5).float()

                dice1 = losses.compute_dice(outputs_1, label_batch)

                dice_scores_1.append(dice1.item())


        avg_dice_1 = sum(dice_scores_1) / len(dice_scores_1)



        logging.info('last best validation dice average: %f ;' % (best_dice))
        logging.info('new validation dice average : %f;' % ( avg_dice_1))



        if  avg_dice_1  > best_dice:
            best_dice =  avg_dice_1
            save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(epoch_num, best_dice))
            torch.save(model_1.state_dict(), save_mode_path)
            logging.info("save best model to {}".format(save_mode_path))



        writer.add_scalar('Val_Dice', avg_dice_1, epoch_num)
        writer.add_scalar('Val_Best_dice', best_dice, epoch_num)



    writer.close()

