
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DoubleConv3D(nn.Module):
    """
    3D U-Net中的双卷积模块：Conv3D -> ReLU -> Conv3D -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, has_dropout = True):
        """
        3D U-Net 构造函数
        :param in_channels: 输入通道数（例如 1 表示灰度图像）
        :param out_channels: 输出通道数（例如分割类别数）
        :param base_channels: 基础通道数
        """
        super(UNet3D, self).__init__()

        self.has_dropout = has_dropout

        # 编码器（下采样）
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)

        # 最大池化层
        self.pool = nn.MaxPool3d(2)

        # 中间层（瓶颈）
        self.bottleneck = DoubleConv3D(base_channels * 8, base_channels * 16)

        # 解码器（上采样）
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        # 最后一层输出
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)  # [B, C, D, H, W]
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))
        if self.has_dropout:
            bottleneck = self.dropout(bottleneck)

        # 解码器
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 跳跃连接
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接
        dec1 = self.dec1(dec1)

        if self.has_dropout:
            dec1 = self.dropout(dec1)

        # 最后一层输出
        out = self.final_conv(dec1)

        return out








class ResidualBlock(nn.Module):
    """
    定义 3D 残差块 (Residual Block)
    包括两个 3D 卷积层，每一层后接 BatchNorm 和 ReLU。
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 如果输入通道和输出通道不同，则需要 1x1 卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + shortcut)  # 残差连接


class Residual3DUNet(nn.Module):
    """
    Residual 3D U-Net 实现
    """

    def __init__(self, in_channels, out_channels, base_channels=32, has_dropout=True, has_proj=False):
        """
        :param in_channels: 输入通道数 (例如 1 表示单通道 CT/MRI 数据)
        :param out_channels: 输出通道数 (例如类别数)
        :param base_channels: U-Net 中的基础通道数
        """
        super(Residual3DUNet, self).__init__()
        self.has_dropout = has_dropout
        self.has_proj = has_proj

        # 编码器 (下采样)
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8)

        # 中间层 (瓶颈层)
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)



        # 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)

        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        # 编码器 (下采样)
        enc1 = self.enc1(x)  # 输入 -> 第一层编码
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2))  # 下采样
        enc3 = self.enc3(F.max_pool3d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool3d(enc3, kernel_size=2))

        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2))
        if self.has_dropout:
            bottleneck = self.dropout(bottleneck)

        # 解码器 (上采样)
        dec4 = self.upconv4(bottleneck)  # 上采样
        dec4 = torch.cat((enc4, dec4), dim=1)  # 跳跃连接 (Skip Connection)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        if self.has_dropout:
            dec1 = self.dropout(dec1)

        # 输出层
        out = self.out_conv(dec1)

        if self.has_proj:
            return out,dec1

        return out


class UNet3D_CSSML(nn.Module):
    """
    Residual 3D U-Net 实现
    """

    def __init__(self, in_channels, out_channels, base_channels=32, has_dropout=True, has_proj=False):
        """
        :param in_channels: 输入通道数 (例如 1 表示单通道 CT/MRI 数据)
        :param out_channels: 输出通道数 (例如类别数)
        :param base_channels: U-Net 中的基础通道数
        """
        super(UNet3D_CSSML, self).__init__()
        self.has_dropout = has_dropout
        self.has_proj = has_proj

        # 编码器 (下采样)
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv3D(base_channels * 4, base_channels * 8)

        # 中间层 (瓶颈层)
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)



        # 解码器 (上采样)
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # self.fc1 = nn.Linear(32, 64)
        # self.fc2 = nn.Linear(64, 16)
        # self.relu = nn.ReLU()
        self.fc_conv = nn.Conv3d(32, 16, kernel_size=1)

    def forward(self, x):
        # 编码器 (下采样)
        enc1 = self.enc1(x)  # 输入 -> 第一层编码
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2))  # 下采样
        enc3 = self.enc3(F.max_pool3d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool3d(enc3, kernel_size=2))

        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2))
        if self.has_dropout:
            bottleneck = self.dropout(bottleneck)

        # 解码器 (上采样)
        dec4 = self.upconv4(bottleneck)  # 上采样
        dec4 = torch.cat((enc4, dec4), dim=1)  # 跳跃连接 (Skip Connection)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        if self.has_dropout:
            dec1 = self.dropout(dec1)

        # 输出层
        out = self.out_conv(dec1)

        if self.has_proj:
            features_in=self.fc_conv(dec1)
            return out, features_in

        return out

if __name__ == '__main__':
    model=UNet3D(in_channels=1, out_channels=2,has_dropout=True)
    model=model.cuda()
    input_data = torch.randn(1, 1, 64, 64, 64)
    input_data = input_data.cuda()

    flops, params = profile(model, inputs=(input_data,))
    print('params:{:.2f}'.format(2 * params / 10e6))
    print('flops:{:.2f}'.format(2 * flops / 10e9))

    # summary(model, (1,1,128,128,64))

    # net = UNet3D(residual='pool')
    # x = torch.ones(4, 1, 128, 128, 64)
    # print (net(x).size())