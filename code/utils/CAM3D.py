import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from monai.networks.nets import VNet
import nibabel as nib


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []

        # 注册钩子
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_tensor, target_class=1):
        # 清除缓存
        self.activations = []
        self.gradients = []

        # 前向传播
        "[1,2,112,112,80]"
        output = self.model(input_tensor)
        if len(output)>1:
            output=output[0]

        pred = torch.argmax(output, dim=1)[0]

        # 反向传播
        self.model.zero_grad()
        loss = output[0, target_class].mean()  # 3D全局平均
        loss.backward()

        # 计算权重
        activation = self.activations[0].squeeze(0)  # [C, D, H, W]
        gradient = self.gradients[0].squeeze(0)  # [C, D, H, W]
        weights = torch.mean(gradient, dim=(1, 2, 3))  # [C,]

        # 3D热力图计算
        cam = torch.einsum('c,cdhw->dhw', weights, activation)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return pred.cpu().numpy(), cam.cpu().numpy()