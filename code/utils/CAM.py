import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from monai.networks.nets import VNet
import nibabel as nib


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = []
        self.gradients = []

        target_layer.register_forward_hook(self.save_feature_map)
        target_layer.register_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        self.feature_maps.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_tensor, target_class=1):
        # 前向传播
        output = self.model(input_tensor)
        output_soft = torch.softmax(output, dim=1)
        prediction = torch.argmax(output_soft, dim=1)
        prediction = prediction[0]

        # 聚合为标量（取目标类别所有像素的均值）
        target = output[:, target_class].mean()  # 关键修改

        # 反向传播
        self.model.zero_grad()
        target.backward()

        # 计算热力图
        feature_map = self.feature_maps[0].squeeze(0)
        gradient = self.gradients[0].squeeze(0)
        weights = torch.mean(gradient, dim=(1, 2, 3), keepdim=True)
        grad_cam = torch.sum(weights * feature_map, dim=0)
        grad_cam = torch.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

        return prediction.cpu().numpy(), grad_cam.cpu().numpy()