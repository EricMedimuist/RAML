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
from medpy.metric.binary import dc

from utils import ramps, losses
from dataloaders.isles2022_process import *


#LOSS
kl_distance = nn.KLDivLoss(reduction='none')
ce_loss = nn.CrossEntropyLoss(reduction='none')
mse_loss = nn.MSELoss(reduction='none')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='ISLES2022', help='dataset_name')
parser.add_argument('--root_path', type=str, default='dataset', help='Name of Experiment dataset')
parser.add_argument('--exp', type=str,  default="RAML", help='reserved_model_name')
parser.add_argument('--max_epochs', type=int,  default=500, help='maximum epoch number to train')
parser.add_argument('--warmup_epochs', type=int,  default=50, help='maximum epoch number to warmup')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_samples', type=int, default=40, help='labeled samples for training')
parser.add_argument('--total_train_samples', type=int, default=200, help='total samples for training')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='initial learning rate')
parser.add_argument('--patch_size', type=float,  default=(64,64,64), help='input image size')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--urr_weight', type=float,  default=0.1, help='uar_weight')
parser.add_argument('--urr_threshold', type=float,  default=0.85, help='uar_threshold')
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.6, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--consistency_change_epoch', type=int, default=20, help='maximum epoch for consistency weight to change')

# PL
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
# CL
parser.add_argument('--CL_type', type=str,default='both', help='CL implement type')
args = parser.parse_args()

"数据集根路径"
train_data_path = args.root_path
"模型权重保存根路径"
snapshot_path = "model/{}/{}_{}_labeled/".format(args.dataset_name, args.exp, args.labeled_samples)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

"训练参数设置"
max_epochs = args.max_epochs
batch_size = args.batch_size * len(args.gpu.split(','))
labeled_bs = args.labeled_bs
base_lr = args.base_lr
warmup_epochs=args.warmup_epochs
patch_size = args.patch_size

num_classes = 2
best_dice_1 = 0
best_dice_2 = 0



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def entropy_map(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    return y1

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)



if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    else:
        shutil.rmtree(snapshot_path)
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    logging.info(str(args))

    writer = SummaryWriter(snapshot_path + '/log')


    model_1 = create_model(name='FEDDNET', in_chns=1, class_num=2,has_proj=True)




    db_train = ISLES2022(base_dir=train_data_path,train_samples_num=args.total_train_samples, split='train', transform_1=get_train_transform_1())
    db_val = ISLES2022(base_dir=train_data_path, split='test', transform_1=get_val_transform())
    logging.info("total train samples: {} ;total test samples: {};".format(len(db_train), len(db_val)))


    labeled_idxs = list(range(args.labeled_samples))           # todo set labeled num
    unlabeled_idxs = list(range(args.labeled_samples, args.total_train_samples))     # todo set unlabeled num
    logging.info("labeled samples: {} ;unlabeled samples: {} ;".format(len(labeled_idxs),len(unlabeled_idxs)))


    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    logging.info("batch_size: {} ;labeled_bs: {};unlabeled_bs: {}".format(batch_size,labeled_bs,batch_size - labeled_bs))

    # batch_sampler1 = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 10, 10-2)
    # for _ in range(1):
    #     i = 0
    #     for x in batch_sampler1:
    #         i += 1
    #         print('%02d' % i, '\t', x)



    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val,  batch_size=1,num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    logging.info(" itertations per epoch of trainloader: {} ;itertations per epoch of valloader: {} ".format(len(trainloader),len(val_loader)))

    # for data in trainloader:
    #     print("imgs:",data[1]['image'][:].size())
    #     print("label:", data[1]['label'][:].size())



    model_1_optimizer = optim.Adam(
        params=model_1.parameters(),  # 要优化的参数
        lr=base_lr,  # 学习率
        betas=(0.9, 0.999),  # 动量参数
        eps=1e-8,  # 防止除零错误的常数
        weight_decay=0.01,  # L2 正则化（权重衰减）
        amsgrad=False  # 是否使用 AMSGrad
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(model_1_optimizer, T_max=(max_epochs - warmup_epochs), eta_min=1e-5)





    iter_num = 0
    lr_ = base_lr
    consistency_weight=0

    "-------------------------训练部分----------------------------"
    for epoch_num in tqdm(range(1,max_epochs+1), ncols=70):
        print('\n')
        time1 = time.time()
        model_1.train()


        supervised_loss_epoch_1=0
        supervised_loss_epoch_2=0

        consistency_loss_epoch_1 = 0
        consistency_loss_epoch_2 = 0


        FC_loss_epoch = 0


        consistency_weight = get_current_consistency_weight(epoch_num // args.consistency_change_epoch)




        # 注意enumerate返回值有两个,一个是训练批次序号，一个是数据（包含训练数据和标签）
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            # print('epoch:{},i_batch:{}'.format(epoch_num,i_batch))



            volume_batch, volume_label = sampled_batch['image'].as_tensor(), sampled_batch['label'].as_tensor()
            volume_label = torch.squeeze(volume_label, dim=1)
            input_batch,label_batch = volume_batch.cuda(), volume_label.cuda()


            outputs_1,outputs_2, feature_out_1, feature_out_2  = model_1(input_batch)


            outputs_1_soft = F.softmax( outputs_1, dim=1)
            outputs_2_soft = F.softmax( outputs_2, dim=1)
            outputs_avg_soft = (outputs_1_soft + outputs_2_soft) / 2



            ## calculate the supervised loss
            "-------------------------------------监督部分-----------------------------------------"
            loss_seg_1 = F.cross_entropy(outputs_1[:labeled_bs], label_batch[:labeled_bs].long())
            loss_seg_dice_1 = losses.dice_loss(outputs_1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            loss_seg_2 = F.cross_entropy(outputs_2[:labeled_bs], label_batch[:labeled_bs].long())
            loss_seg_dice_2 = losses.dice_loss(outputs_2_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)


            "-------------------------------------URR-----------------------------------------"

            outputs_1_onehot = torch.max(outputs_1_soft[:labeled_bs, :, :, :, :], dim=1)[1]
            # outputs_1_onehot = torch.argmax(outputs_1_soft[:labeled_bs, :, :, :, :], dim=1)
            outputs_2_onehot = torch.max(outputs_2_soft[:labeled_bs, :, :, :, :], dim=1)[1]

            uncertainty_avg = -1.0 * torch.sum(outputs_avg_soft[:labeled_bs, ...] * torch.log(outputs_avg_soft[:labeled_bs, ...] + 1e-6), dim=1)

            ent_mean =  uncertainty_avg.mean()
            ent_max = uncertainty_avg.max()


            # ent_threshold = ent_mean + (0.8+0.15*(epoch_num/max_epochs))*(ent_max - ent_mean)
            ent_threshold = ent_mean + (0.5 + (args.uar_threshold-0.5) * (1 / (1 + np.exp(-0.05 * (epoch_num - 0.1*max_epochs))))) * (ent_max - ent_mean)
            # ent_threshold = ent_mean + 0.5 * (ent_max - ent_mean)


            unreliable_mask = ( uncertainty_avg > ent_threshold ).to(torch.int32)




            diff_dist_1 = mse_loss(outputs_1_soft[:labeled_bs,1,...], label_batch[:labeled_bs])
            diff_dist_2 = mse_loss(outputs_2_soft[:labeled_bs,1,...], label_batch[:labeled_bs])


            diff_loss_1 = torch.sum(unreliable_mask * diff_dist_1) / (torch.sum(unreliable_mask) + 1e-16)
            diff_loss_2 = torch.sum(unreliable_mask * diff_dist_2) / (torch.sum(unreliable_mask) + 1e-16)

            "total supervised loss for this batch"
            supervised_loss_1 = 1 * (loss_seg_1 + loss_seg_dice_1) + args.uar_weight * diff_loss_1
            supervised_loss_2 = 1 * (loss_seg_2 + loss_seg_dice_2) + args.uar_weight * diff_loss_2

            "total supervised loss for this epoch"
            supervised_loss_epoch_1 += supervised_loss_1
            supervised_loss_epoch_2 += supervised_loss_2



            "-------------------------------------无监督部分-----------------------------------------"
            "-------------------------------------特征一致性模块（FDL）-----------------------------------------"

            feature_flat_1 = feature_out_1.view(-1, feature_out_1.size(1))
            feature_flat_2 = feature_out_2.view(-1, feature_out_2.size(1))


            similarity =  F.cosine_similarity(feature_flat_1.detach(), feature_flat_2, dim=1)
            FC_loss = 1 + torch.mean(similarity)

            FC_loss_epoch += FC_loss

            "-------------------------------------交叉伪监督模块-----------------------------------------"


            outputs_1_unlabeled = outputs_1[labeled_bs:, :, :, :, :]
            outputs_2_unlabeled = outputs_2[labeled_bs:, :, :, :, :]

            outputs_1_soft_unlabeled = outputs_1_soft[labeled_bs:, :, :, :, :]
            outputs_2_soft_unlabeled = outputs_2_soft[labeled_bs:, :, :, :, :]


            outputs_PLable_1 = torch.argmax( outputs_1_soft_unlabeled.detach(), dim=1)
            outputs_PLable_2 = torch.argmax( outputs_2_soft_unlabeled.detach(), dim=1)

            KL_map = torch.sum(kl_distance(torch.log(outputs_1_soft_unlabeled), outputs_2_soft_unlabeled.detach()), dim=1)
            # print("KL_map:",KL_map.shape)


            KL_weight_map = torch.exp(-1 * KL_map.detach())


            consistency_dist_2to1 = ce_loss(outputs_1_unlabeled, outputs_PLable_2)
            consistency_loss_2to1 = torch.mean(KL_weight_map * consistency_dist_2to1)
            # consistency_loss_2to1 = torch.mean(KL_weight_map * consistency_dist_2to1) + torch.mean(KL_map)

            consistency_dist_1to2 = ce_loss(outputs_2_unlabeled, outputs_PLable_1)
            consistency_loss_1to2 = torch.mean(KL_weight_map * consistency_dist_1to2)
            # consistency_loss_1to2 = torch.mean(KL_weight_map * consistency_dist_1to2) + torch.mean(KL_map)




            "total consistency loss for this batch"
            consistency_loss_1 = consistency_loss_2to1
            consistency_loss_2 = consistency_loss_1to2

            "total consistency loss for this epoch"
            consistency_loss_epoch_1 += consistency_loss_1
            consistency_loss_epoch_2 += consistency_loss_2


            loss_decoder_1 = supervised_loss_1 + consistency_weight * ( consistency_loss_1)
            loss_decoder_2 = supervised_loss_2 + consistency_weight * ( consistency_loss_2)
            loss_1_total = loss_decoder_1 + loss_decoder_2 +  consistency_weight * FC_loss



            model_1_optimizer.zero_grad()
            loss_1_total.backward()
            model_1_optimizer.step()


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
        "记录每一轮的一致性权重"
        logging.info('consistency_weight: %f ;' % (consistency_weight))

        "记录每一轮的平均监督损失"
        supervised_loss_epoch_1 = supervised_loss_epoch_1 / len(trainloader)
        supervised_loss_epoch_2 = supervised_loss_epoch_2 / len(trainloader)
        logging.info(
            'supervised_loss_epoch_1: %f ;supervised_loss_epoch_2: %f ; '
            % ( supervised_loss_epoch_1,  supervised_loss_epoch_2))

        "记录每一轮的平均伪监督损失"
        consistency_loss_epoch_1 = consistency_loss_epoch_1/ len(trainloader)
        consistency_loss_epoch_2 = consistency_loss_epoch_2 / len(trainloader)

        logging.info(
            'consistency_loss_epoch_1: %f ;consistency_loss_epoch_2: %f ; '
            % (consistency_loss_epoch_1, consistency_loss_epoch_2))

        "记录每一轮的特征一致性损失"
        FC_loss_epoch = FC_loss_epoch / len(trainloader)
        logging.info(
            'FC_loss_epoch: %f ; ' % (FC_loss_epoch))

        "画每一轮的学习率，一致性权重，总损失，监督损失，伪监督损失"
        writer.add_scalar('learning_rate',  model_1_optimizer.state_dict()['param_groups'][0]['lr'], epoch_num)
        writer.add_scalar('consistency_weight',  consistency_weight, epoch_num)
        writer.add_scalar('loss/supervised_loss_1', supervised_loss_epoch_1.item(), epoch_num)
        writer.add_scalar('loss/supervised_loss_2', supervised_loss_epoch_2.item(), epoch_num)
        writer.add_scalar('loss/pseudolabel_loss_1', consistency_loss_epoch_1.item(), epoch_num)
        writer.add_scalar('loss/pseudolabel_loss_2', consistency_loss_epoch_2.item(), epoch_num)
        writer.add_scalar('loss/FC_loss', FC_loss_epoch.item(), epoch_num)





        "-----------------------------val----------------------------------"

        model_1.eval()
        dice_scores_1 = []
        dice_scores_2 = []
        dice_socre=[]

        # prompt_dice_list = []
        for i_batch, sampled_batch in enumerate(val_loader):
            volume_batch_val, volume_label_val = sampled_batch['image'].as_tensor(), sampled_batch['label'].as_tensor()
            volume_label_val = torch.squeeze(volume_label_val, dim=1)
            input_batch, label_batch = volume_batch_val.cuda(), volume_label_val.cuda()

            with torch.no_grad():
                outputs_1, outputs_2,_,_ = model_1(input_batch)

                "[4, 2, 64,64,64]"
                outputs_1 = F.softmax(outputs_1, dim=1)
                outputs_2 = F.softmax(outputs_2, dim=1)

                "[4, 64,64,64]"
                outputs_1 = torch.argmax(outputs_1, dim=1)
                outputs_2 = torch.argmax(outputs_2, dim=1)

                "[4, 64,64,64]"
                label_batch = (label_batch > 0.5).float()

                dice1 = losses.compute_dice(outputs_1, label_batch)
                dice2 = losses.compute_dice(outputs_2, label_batch)

                outputs_1 = outputs_1.detach().cpu().numpy()
                label_batch = label_batch.detach().cpu().numpy()
                dice = dc(outputs_1, label_batch)

                dice_scores_1.append(dice1.item())
                dice_scores_2.append(dice2.item())
                dice_socre.append(dice)



        avg_dice_1 = sum(dice_scores_1) / len(dice_scores_1)
        avg_dice_2 = sum(dice_scores_2) / len(dice_scores_2)

        avg_dice = sum(dice_socre) / len(dice_socre)


        logging.info('last best validation dice average for network_1,2: %f,%f ;' % (best_dice_1, best_dice_2))
        logging.info('new validation dice average for network_1,2: %f, %f;' % (avg_dice_1, avg_dice_2))
        logging.info('new validation dice average for network_1: %f;' % (avg_dice))


        if avg_dice_1 > best_dice_1:
            best_dice_1 = avg_dice_1
            logging.info('!!!!!!!best validation dice  for network_1 has changed to: %f !!!!!!!;' % (best_dice_1))

            save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(epoch_num, best_dice_1))
            torch.save(model_1.state_dict(), save_mode_path)
            logging.info("save best model to {}".format(save_mode_path))

        if avg_dice_2 > best_dice_2:
            best_dice_2 = avg_dice_2
            logging.info('!!!!!!!best validation dice  for network_2 has changed to: %f !!!!!!!;' % (best_dice_2))


        writer.add_scalar('Val_Dice_1', avg_dice_1, epoch_num)
        writer.add_scalar('Val_Best_dice_1', best_dice_1, epoch_num)
        writer.add_scalar('Val_Dice_2', avg_dice_2, epoch_num)
        writer.add_scalar('Val_Best_dice_2', best_dice_2, epoch_num)




    writer.close()

