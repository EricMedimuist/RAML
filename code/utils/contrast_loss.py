'''
Date: 2023-10-18 22:10:13
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-18 22:58:47
FilePath: /CAC4SSL/code/utils/contrast_loss.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast(nn.Module):

    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super(Contrast, self).__init__()
        self.tau = temperature

    def forward(self, proj_list, idx,  mask, sample_num=5):
        batch_size = mask.shape[0]
        loss = 0
        
        curr_proj = None
        pos_proj = []
        for i in range(len(proj_list)):
            try:
                proj = proj_list[i].permute(0, 2, 3, 1)
            except:
                "[ 4, 64,64,64, 16]"
                proj = proj_list[i].permute(0, 2, 3, 4, 1)

            "调整维度，[ 4, 64x64x64, 16]"
            proj = proj.reshape(proj.shape[0], -1, proj.shape[-1])
            # print("proj:", proj.shape)

            "投影输出在通道维度上归一化"
            if i == idx:
                "给定投影，[ 4, 64x64x64, 16]"
                curr_proj = F.normalize(proj, dim=-1)
                # print("  curr_proj:",  curr_proj.shape)
            else:
                "其他投影，[ 4, 1, 64x64x64, 16]"
                pos_proj.append(F.normalize(proj.unsqueeze(1), dim=-1))

        "其他投影集合，[ 4, 1, 64x64x64, 16]"
        pos_proj = torch.cat(pos_proj, dim=1)
        # print(" pos_proj:", pos_proj.shape)

        "掩码，[4, 64x64x64]"
        mask = mask.reshape(batch_size, -1).long()
        # print(" mask:", mask)
        # print(" mask:",mask.shape)
        "模糊图掩码"
        fn_mask = 1 - mask

        "对于这一批次内的每一个特征图"
        for b_idx in range(batch_size):

            "不确定体素的掩码，[ 64x64x64]"
            fn_mask_ = fn_mask[b_idx]
            # print("fn_mask_:", fn_mask_)
            "给定投影，[64x64x64, 16]"
            c_proj = curr_proj[b_idx]
            # print(" c_proj.shape:", c_proj.shape)
            "其他投影，[1,64x64x64, 16]"
            p_proj = pos_proj[b_idx]
            # print(" p_proj.shape:", p_proj.shape)

            "记录fn_mask_中非零元素（不确定体素）下标"
            hard_indices = fn_mask_.nonzero()
            # print("hard_indices:",hard_indices)
            # print("hard_indices.shape():", hard_indices.shape)
            # print("fn_mask_.sum():", fn_mask_.sum())

            "num_hard被赋值为fn_mask_中发现的非零元素（不确定体素）的数量"
            num_hard = hard_indices.shape[0]
            # print("num_hard:", num_hard)

            "在不确定体素中取sample_num个"
            hard_sample_num = min(sample_num, num_hard)
            # print("hard_sample_num :", hard_sample_num )

            "根据num_hard建立索引,将0~n-1（包括0和n-1）随机打乱后获得的数字序列"
            hard_perm = torch.randperm(num_hard)
            # print(" hard_perm:",  hard_perm)

            "从hard_indices中随机取hard_sample_num个样本的索引"
            indices = hard_indices[hard_perm[:hard_sample_num]]
            # print("hard_perm[:hard_sample_num]:", hard_perm[:hard_sample_num])
            # print("indices:", indices)

            "被挑选出的锚点特征投影，从不确定体素区域随机采样了sample_num个点，[sample_num,16]"
            c_proj_selected = c_proj[indices].squeeze(dim=1)
            # print(" c_proj[indices].shape:", c_proj[indices].shape)
            # print(" c_proj_selected:", c_proj_selected.shape)

            "被挑选出的样本特征投影,[1,sample_num,16]"
            p_proj_selected = p_proj[:, indices].squeeze(dim=2)
            # print(" p_proj[:, indices]:", p_proj[:, indices].shape)
            # print(" p_proj_selected:", p_proj_selected.shape)

            "正样本相似度,[5,16];[1,5,16]"
            pos_loss_item = F.cosine_similarity(c_proj_selected, p_proj_selected, dim=-1)

            # print(" pos_loss_item:", pos_loss_item.shape)
            pos_loss_item = pos_loss_item.sum(0)
            # print(" pos_loss_item:", pos_loss_item.shape)
            pos_loss_item = torch.exp(pos_loss_item / self.tau)
            # print(" pos_loss_item:", pos_loss_item.shape)

            "负样本相似度，计算5个锚点之间的相似度矩阵，[5,1,16] [1,5,16]"
            matrix = F.cosine_similarity(c_proj_selected.unsqueeze(dim=1), c_proj_selected.unsqueeze(dim=0), dim=-1)
            matrix = torch.exp(matrix / self.tau)
            # print("  matrix :",  matrix.shape )
            neg_loss_item = matrix.sum(dim=0) - torch.diagonal(matrix)
            # print("  neg_loss_item :",  neg_loss_item.shape)

            loss += -torch.log(pos_loss_item / (pos_loss_item + neg_loss_item + 1e-8)).mean()

        return loss / batch_size
    