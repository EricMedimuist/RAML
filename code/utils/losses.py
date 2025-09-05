import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_coefficient_sample(pred, target, dim=0, smooth=1e-6):
    dicelist=[]
    for i in range(pred.shape[dim]):
        dice=compute_dice(pred[i], target, smooth=1e-6)
        dicelist.append(dice)
    return (dicelist)


def compute_dice(pred, target, smooth=1e-6):
    # 将预测和目标转换为二进制掩码
    pred = (pred > 0.5).float()  # 假设使用 0.5 作为阈值
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_iou(pred, target):
    # 将预测和目标转换为二进制掩码
    pred = (pred > 0.5).float()  # 假设使用 0.5 作为阈值
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union if union > 0 else 0.0

def compute_accuracy(pred, target):
    # 将预测和目标转换为二进制掩码
    pred = (pred > 0.5).float()  # 假设使用 0.5 作为阈值
    target = (target > 0.5).float()

    correct = (pred == target).sum()
    # print(" correct:", correct)
    total = target.numel()
    # print(" total:", total)

    return correct / total

def compute_recall(pred, target):
    # 将预测和目标转换为二进制掩码
    pred = (pred > 0.5).float()  # 假设使用 0.5 作为阈值
    target = (target > 0.5).float()

    true_positive = (pred * target).sum()
    possible_positive = target.sum()
    return true_positive / possible_positive if possible_positive > 0 else 0.0


def Binary_dice_loss(predictive, target, ep=1e-8, mask=None):
    if mask is not None:
        predictive = torch.masked_select(predictive, mask)
        target = torch.masked_select(target, mask)
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def ce_loss_mask(input1, input2, mask):
    loss_f = nn.CrossEntropyLoss(reduction='none')
    loss = loss_f(input1, input2) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def mse_loss(input1, input2, mask=None):
    if mask is None:
        return torch.mean((input1 - input2)**2)
    else:
        mse = (input1 - input2)**2
        return torch.mean(torch.masked_select(mse, mask.bool().unsqueeze(1)))

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()

    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
