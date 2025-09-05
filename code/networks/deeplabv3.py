import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from thop import profile


class Conv3dBNReLU(nn.Module):
    """3D卷积 + BN + ReLU基础模块"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling模块 - 修复版本"""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()

        # 1x1x1卷积
        self.conv1x1 = Conv3dBNReLU(in_channels, out_channels, kernel_size=1)

        # 3x3x3卷积，不同膨胀率 - 使用合适的padding保持尺寸
        self.conv3x3_1 = Conv3dBNReLU(in_channels, out_channels, kernel_size=3,
                                      padding=1, dilation=1)  # 使用较小的膨胀率
        self.conv3x3_2 = Conv3dBNReLU(in_channels, out_channels, kernel_size=3,
                                      padding=2, dilation=2)  # 调整padding
        self.conv3x3_3 = Conv3dBNReLU(in_channels, out_channels, kernel_size=3,
                                      padding=3, dilation=3)  # 调整padding

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Conv3dBNReLU(in_channels, out_channels, kernel_size=1)
        )

        # 融合卷积
        self.fusion = nn.Sequential(
            Conv3dBNReLU(5 * out_channels, out_channels, kernel_size=1),
            nn.Dropout3d(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()

        # 各个分支
        feat1x1 = self.conv1x1(x)
        feat3x3_1 = self.conv3x3_1(x)
        feat3x3_2 = self.conv3x3_2(x)
        feat3x3_3 = self.conv3x3_3(x)

        # 全局平均池化分支
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(d, h, w),
                                    mode='trilinear', align_corners=True)

        # 确保所有特征图尺寸一致
        if feat3x3_1.shape[2:] != (d, h, w):
            feat3x3_1 = F.interpolate(feat3x3_1, size=(d, h, w),
                                      mode='trilinear', align_corners=True)
        if feat3x3_2.shape[2:] != (d, h, w):
            feat3x3_2 = F.interpolate(feat3x3_2, size=(d, h, w),
                                      mode='trilinear', align_corners=True)
        if feat3x3_3.shape[2:] != (d, h, w):
            feat3x3_3 = F.interpolate(feat3x3_3, size=(d, h, w),
                                      mode='trilinear', align_corners=True)

        # 拼接所有特征
        combined = torch.cat([feat1x1, feat3x3_1, feat3x3_2, feat3x3_3, global_feat], dim=1)

        return self.fusion(combined)


class ResNet3DBasicBlock(nn.Module):
    """3D ResNet基础块"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv3dBNReLU(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        # 下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet3DEncoder(nn.Module):
    """3D ResNet编码器"""

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 四个残差块阶段
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        # 输出通道数
        self.out_channels = base_channels * 8

    def _make_layer(self, in_channels: int, out_channels: int,
                    blocks: int, stride: int = 1) -> nn.Module:
        layers = []
        layers.append(ResNet3DBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNet3DBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 初始特征提取
        x = self.conv1(x)  # 1/2
        x = self.maxpool(x)  # 1/4

        # 各个阶段
        low_level_feat = self.layer1(x)  # 1/4 - 用作低级特征
        x = self.layer2(low_level_feat)  # 1/8
        x = self.layer3(x)  # 1/16
        x = self.layer4(x)  # 1/32 - 高级语义特征

        return x, low_level_feat


class DeepLabV3Plus3DDecoder(nn.Module):
    """3D DeepLabV3+解码器"""

    def __init__(self, base_channels: int, aspp_channels: int, num_classes: int):
        super().__init__()

        # 低级特征处理
        self.low_level_conv = Conv3dBNReLU(base_channels, 48, kernel_size=1)

        # 高级特征上采样
        self.high_level_conv = nn.Sequential(
            Conv3dBNReLU(aspp_channels, 256, kernel_size=3, padding=1),
            nn.Dropout3d(0.5)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            Conv3dBNReLU(256 + 48, 256, kernel_size=3, padding=1),
            nn.Dropout3d(0.5),
            Conv3dBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Dropout3d(0.1)
        )

        # 最终分类层
        self.classifier = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, high_level_feat: torch.Tensor,
                low_level_feat: torch.Tensor) -> torch.Tensor:
        # 处理低级特征
        low_level_feat = self.low_level_conv(low_level_feat)

        # 处理高级特征并上采样
        high_level_feat = self.high_level_conv(high_level_feat)
        high_level_feat = F.interpolate(high_level_feat,
                                        size=low_level_feat.shape[2:],
                                        mode='trilinear',
                                        align_corners=True)

        # 特征拼接
        combined = torch.cat([high_level_feat, low_level_feat], dim=1)

        # 融合特征
        fused = self.fusion_conv(combined)

        # 最终上采样到原始尺寸
        output = self.classifier(fused)
        output = F.interpolate(output, scale_factor=4, mode='trilinear', align_corners=True)

        return output


class DeepLabV3Plus3D(nn.Module):
    """3D DeepLabV3+完整网络 - 修复版本"""

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_channels: int = 64, aspp_channels: int = 256):
        super().__init__()

        # 编码器
        self.encoder = ResNet3DEncoder(in_channels, base_channels)

        # ASPP模块 - 使用修复版本
        self.aspp = ASPP3D(self.encoder.out_channels, aspp_channels)

        # 解码器
        self.decoder = DeepLabV3Plus3DDecoder(base_channels, aspp_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器前向传播
        high_level_feat, low_level_feat = self.encoder(x)

        # ASPP处理高级特征
        aspp_feat = self.aspp(high_level_feat)

        # 解码器
        output = self.decoder(aspp_feat, low_level_feat)

        return output


# 测试代码
def test_deeplabv3plus_3d():
    """测试修复后的3D DeepLabV3+网络"""
    print("Testing Fixed DeepLabV3+ 3D...")

    # 创建模型
    model = DeepLabV3Plus3D(in_channels=1, num_classes=2, base_channels=32)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    input_data = torch.randn(1, 1, 64, 64, 64)

    try:
        with torch.no_grad():
            output = model(input_data)
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")

        # 测试FLOPs计算
        flops, params = profile(model, inputs=(input_data,))
        print('Params: {:.2f} M'.format(params / 1e6))
        print('FLOPs: {:.2f} G'.format(flops / 1e9))

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # 运行测试
    success = test_deeplabv3plus_3d()
