"-----------MCNet3d---------"
import torch
from torch import nn
import torch.nn.functional as F
from thop import profile
from torchstat import stat
from torchsummary import summary
class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class MultiPath3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MultiPath3DConv, self).__init__()


        # 路径1: 单卷积 + 残差
        self.path1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 路径2: 双卷积 + 残差
        self.path2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

        )

        # 路径3: 三卷积 + 残差
        self.path3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

        )
        # 路径4: 直连路径 (identity)
        self.path4 = nn.Identity()

        # 最终融合卷积
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x.shape)
        # 并行计算各路径
        p1 = self.path1(x)
        # print(p1.shape)
        p2 = self.path2(x)
        p3 = self.path3(x)
        p4 = self.path4(x)  # 恒等映射

        # 拼接各路径结果
        out = p1 +  p2 + p3 +p4

        # # 融合特征
        # out = self.final_conv(out)
        # out = self.bn(out)
        # out = self.relu(out)
        return out
"步长卷积下采样"
class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

"maxpool下采样，卷积调整通道数"
class DownsamplingMaxpoolBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out,  normalization='none'):
        super(DownsamplingMaxpoolBlock, self).__init__()

        ops = []
        ops.append(nn.MaxPool3d(kernel_size=(2, 2, 2),stride=2))
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

"转置卷积上采样"
class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

"三线性插值上采样+卷积调整通道数"
class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



"通道注意力"
class ChannelAttention(nn.Module):
    """
    Channel Attention Module

    As described in https://arxiv.org/abs/1807.06521.
    """

    def  __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

        # Define the shared layers
        self.shared_layer_one = nn.Linear(channel, channel // self.ratio)
        self.shared_layer_two = nn.Linear(channel // self.ratio, channel)

    def forward(self, x):
        batch_size, channel, _, _, _ = x.size()

        # Average pooling
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(batch_size, channel)
        avg_pool = F.relu(self.shared_layer_one(avg_pool))
        avg_pool = self.shared_layer_two(avg_pool).view(batch_size, channel, 1, 1, 1)

        # Max pooling
        max_pool = F.adaptive_max_pool3d(x, 1).view(batch_size, channel)
        max_pool = F.relu(self.shared_layer_one(max_pool))
        max_pool =self.shared_layer_two(max_pool).view(batch_size, channel, 1, 1, 1)

        # Combine the two features
        feature = avg_pool + max_pool
        feature = torch.sigmoid(feature)
        # Scale the input
        return x * feature + x


"空间注意力"
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=self.kernel_size,
                                stride=1, padding=self.kernel_size // 2, bias=False)

    def forward(self, x):
        # Average pooling along the channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along the channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]

        # Concatenate along the channel dimension
        concat = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolution to get the attention feature
        feature = torch.sigmoid(self.conv3d(concat))

        # Scale the input features
        return x * feature + x




class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res



class Decoder(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=True,
                 up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg



"for Mutual Reliable"
class Decoder_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=True,
                 up_type=0):
        super(Decoder_v1, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg, x9






class Decoder_FEDDNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True, up_type=0,has_attention=False,has_proj=False):
        super(Decoder_FEDDNet, self).__init__()
        self.has_dropout = has_dropout
        self.has_attention = has_attention
        self.has_proj= has_proj

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = MultiPath3DConv(n_filters * 8, n_filters * 8, kernel_size=3, stride=1, padding=1)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = MultiPath3DConv(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding=1)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.attention_Channel_three = ChannelAttention(channel=n_filters * 4, ratio=8)
        self.attention_Channel_four = ChannelAttention(channel=n_filters * 8, ratio=8)
        self.attention_Channel_five = ChannelAttention(channel=n_filters * 16, ratio=8)
        self.attention_Channel_six = ChannelAttention(channel=n_filters * 8, ratio=8)
        self.attention_Channel_seven = ChannelAttention(channel=n_filters * 4, ratio=8)
        self.attention_Channel_eight = ChannelAttention(channel=n_filters * 2, ratio=8)
        self.attention_Spatial = SpatialAttention(kernel_size=7)


        "特征投影层的设计"
        self.feature_avg_layer = nn.AvgPool3d(kernel_size=3, stride=3, padding=0)
        self.feature_max_layer = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.feature_in_conv = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1)
        self.feature_out_conv = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        if self.has_attention:
            x4 = self.attention_Channel_four(x4)
            x4 = self.attention_Spatial(x4)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        if self.has_attention:
            x3 = self.attention_Channel_three(x3)
            x3 = self.attention_Spatial(x3)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        # if self.has_attention:
        #     x7 = self.attention_Channel_seven(x7)
        #     x7 = self.attention_Spatial(x7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        # if self.has_attention:
        #     x8 = self.attention_Channel_eight(x8)
        #     x8 = self.attention_Spatial(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)


        out_seg = self.out_conv(x9)
        # proj_out = self.proj(x9)


        if self.has_proj:

            avg_feature = self.feature_avg_layer(x9)
            max_feature = self.feature_max_layer(x9)
            features_in = self.feature_in_conv( self.sigmoid(avg_feature+max_feature))
            # print(" features_in:", features_in.shape)
            features_in = features_in.view(-1,features_in.size(1))
            fc1= self.relu(self.fc1(features_in))
            # print(" fc1.shape:",fc1.shape)
            proj_out =self.fc2(fc1)

            return out_seg, proj_out

        return out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout(x, p)
    return x

def FeatureNoise(x, noise_range=0.3):
    noise = torch.clamp(torch.randn_like(x) * 0.1, -noise_range, noise_range)
    x = x + noise
    return x





class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2

class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3



"for Mutual Reliable"
class Mine3d_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=True):
        super(Mine3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2


"不同导向的2解码器"
class MAONet3d_v2(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True,has_proj=False):
        super(MAONet3d_v2, self).__init__()
        self.has_proj = has_proj
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_MAO_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,  up_type=0, has_attention=False,has_proj=has_proj)
        self.decoder2 = Decoder_MAO_v2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,  up_type=0, has_attention=True,has_proj=has_proj)


    def forward(self, input):
        features = self.encoder(input)

        if self.has_proj:
            out_seg1,proj_out1 = self.decoder1(features)
            out_seg2,proj_out2 = self.decoder2(features)
            return out_seg1, out_seg2, proj_out1, proj_out2

        else:
            out_seg1 = self.decoder1(features)
            out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2

class FEDDNET(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=True,has_proj=False):
        super(FEDDNET, self).__init__()
        self.has_proj = has_proj
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_FEDDNet(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,  up_type=0, has_attention=False,has_proj=has_proj)
        self.decoder2 = Decoder_FEDDNet(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,  up_type=0, has_attention=True,has_proj=has_proj)


    def forward(self, input):
        features = self.encoder(input)

        if self.has_proj:
            out_seg1,proj_out1 = self.decoder1(features)
            out_seg2,proj_out2 = self.decoder2(features)
            return out_seg1, out_seg2, proj_out1, proj_out2

        else:
            out_seg1 = self.decoder1(features)
            out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2





if __name__ == "__main__":
    model=FEDDNET(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    model = model.cuda()
    input_data = torch.randn(1, 1, 64, 64, 64)
    input_data = input_data.cuda()


    flops, params = profile(model, inputs=(input_data,))
    print('params:{:.2f}'.format(params / 10e6))
    print('flops:{:.2f}'.format(flops / 10e9))



    # "torch.Size([4, 2, 64, 64, 64])和torch.Size([4, 16, 32, 32, 32])"
    # out_seg1,out_seg2,proj_out1,proj_out2= model(input_data)
    # print(" out_seg1:",  out_seg1.shape)
    # print("proj_out1:",proj_out1.shape)
    # "torch.Size([4x32x32x32, 16])"
    # feature_flat_1 = proj_out1.view(-1,proj_out1.size(1))
    # feature_flat_2 = proj_out2.view(-1, proj_out2.size(1))
    # print("feature_flat_1:", feature_flat_1.shape)
    #
    # # 计算余弦相似度
    # cos_sim = F.cosine_similarity( feature_flat_1, feature_flat_2, dim=1)
    # print(" cos_sim:", cos_sim)
    #
    # # 计算一致性损失（1 - 余弦相似度的均值）
    # loss = 1 - cos_sim.mean()
    # print("loss:",loss)
