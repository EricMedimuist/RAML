import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
import  shutil
# from torchvision import transforms

import SimpleITK as sitk
from monai import transforms
from monai.data import Dataset, DataLoader
from monai.apps import CrossValidation
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import json


roi_size=(64, 64, 64)
test_size=(64, 64, 64)

class ISLES2022(Dataset):
    """ ISLES Dataset """
    def __init__(self, base_dir=None, split='train',train_samples_num=200, transform_1=None):
        self._base_dir = base_dir
        self.transform_1= transform_1
        self.sample_list = []
        self.name_list = []
        self.train_samples_num = train_samples_num
        self.test_samples_num = 50
        self.split=split


        if split == 'train':
            for i in range(1,train_samples_num+1):
                self.name_list.append(str(i))
        elif  split == 'test':
            for i in range(200+1,200+self.test_samples_num+1):
                self.name_list.append(str(i))

        # print("image_list:",self.name_list)
        # print("total {} samples".format(len(self.name_list)))

    def __len__(self):
        return  len(self.name_list)

    def __getitem__(self, idx):
        "根据idx索引读取文件"

        data_path=self._base_dir+"/ISLES_2022/Sample_"+ self.name_list[idx]
        img_nib_path = os.path.join(data_path,'dwi_case.nii.gz')
        label_nib_path =os.path.join(data_path, 'label_case.nii.gz')

        sample = {'image': img_nib_path, 'label': label_nib_path }   #获取图像标签字典
        # print("sample",sample)


        if self.transform_1:
            sample = self.transform_1(sample)


        return  sample




def get_train_transform_1():
    # 定义转换
    train_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 裁剪前景区域


        transforms.RandSpatialCropd(
            ["image", "label"], roi_size=roi_size, random_size=False
        ),
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=roi_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),

        ),

        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        transforms.RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),  # 随机缩放
        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),

    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform



def get_val_transform():
    val_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 裁剪前景区域


        transforms.Resized(["image", "label"], spatial_size=test_size, size_mode='all', mode="bilinear", align_corners=False),
        # transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # 强度归一化到 [0, 1]
        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),  # 确保数据为 Tensor 格式

    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform


def get_test_transform():
    test_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 裁剪前景区域

        transforms.Resized(["image", "label"], spatial_size=test_size, size_mode='all', mode="bilinear",
                           align_corners=False),

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),  # 确保数据为 Tensor 格式

    ]
    test_transform = transforms.Compose(test_transform)
    return test_transform


def get_CAM_val_transform():
    val_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 裁剪前景区域

        transforms.Resized(["image", "label"], spatial_size=(96,96,96), size_mode='all', mode="bilinear",
                           align_corners=False),

        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),  # 确保数据为 Tensor 格式

    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform

def get_CAM_original_transform():
    test_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),  # 裁剪前景区域

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),  # 确保数据为 Tensor 格式

    ]
    test_transform = transforms.Compose(test_transform)
    return test_transform



def convert():

    directory = '../../dataset'
    for i in range(1, 251):
        new_folder = os.path.join(directory, 'ISLES_2022/Sample_' + str(i))
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    for i in range(1,251):
        number_zero=4-len(str(i))
        string='0'*number_zero+str(i)
        print(string)
        img_source = os.path.join(directory,
                                  'ISLES_2022_raw/sub-strokecase'+string+'/ses-0001/dwi/sub-strokecase'+string+'_ses-0001_dwi.nii.gz')
        img_destiny = os.path.join(directory, 'ISLES_2022/Sample_'+str(i)+'/dwi_case.nii.gz')
        shutil.copy(img_source, img_destiny)

        label_source = os.path.join(directory, 'ISLES_2022_raw/derivatives/sub-strokecase'+string+'/ses-0001/sub-strokecase'+string+'_ses-0001_msk.nii.gz')
        label_destiny = os.path.join(directory, 'ISLES_2022/Sample_'+str(i)+'/label_case.nii.gz')
        shutil.copy(label_source, label_destiny)

def get_train_transform_my():
    # 定义转换
    train_transform = [
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True, allow_missing_keys=True),
        #使用边框裁剪图像，确保裁剪到包含前景的范围，source_key：按image还是label裁剪图像
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        #调整空间分辨率
        transforms.Spacingd(
            ["image", "label"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear", "bilinear"),
        ),
        #随机裁剪
        transforms.RandSpatialCropd(
            ["image", "label"], roi_size=(64, 64, 64), random_size=False
        ),
        #仿射变换，spatial_size指定输出图像和标签
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=(64, 64, 64),
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),

        ),
        #沿轴向翻转
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),

        # 随机Gibbs噪声
        transforms.RandGibbsNoise(prob=0.1, alpha=(0.0, 1.0)),
        # 随机高斯噪声
        transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
        # 对图像应用随机的高斯平滑（模糊）处理,sigma控制高斯平滑的标准差
        transforms.RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        #随机应用高斯锐化效果，以增强医学图像的细节和边缘
        transforms.RandGaussianSharpend("image", prob=0.2),
        #随机缩放图像的强度值，factor为缩放因子，v = v * (1 + factor)
        transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
        #随机调整图像的强度值，offsets表示偏移量为（-0.1,0.1）
        transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        #随机标准差平移强度值,v = v + factor * std(v)
        transforms.RandStdShiftIntensityd("image",factors=0.1,  prob=0.15),
        #随机调整图像的对比度,gamma为对比度缩放范围
        transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),

        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)   #0-15
        secondary_iter = iterate_eternally(self.secondary_indices)  #16-79
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def size_range():
    db_train = ISLES2022(base_dir='../../dataset',
                         split='train',
                         # todo change training flod
                         transform_1=get_test_transform(),
                         )

    # print("db_train", db_train[19])
    minsize1 = 1000
    minsize2 = 1000
    minsize3 = 1000

    maxsize1 = 0
    maxsize2 = 0
    maxsize3 = 0
    for i in range(len(db_train)):
        img = db_train[i]['image'].as_tensor()
        label = db_train[i]['label'].as_tensor()
        # print("img:", img)
        # print("label :", label)
        minsize1 = min(minsize1, img.shape[1])
        minsize2 = min(minsize2, img.shape[2])
        minsize3 = min(minsize3, img.shape[3])

        maxsize1 = max(maxsize1, img.shape[1])
        maxsize2 = max(maxsize2, img.shape[2])
        maxsize3 = max(maxsize3, img.shape[3])
        print("img", img.shape)
        print("label", label.shape)
        print("img.max()", img.max())
        print("img.min()", img.min())
        print("img.sum()", img.sum())
        print("label.sum()", label.sum())

    "foreground minsize: 62 73 24"
    print("minsize:", minsize1, minsize2, minsize3)
    "foreground maxsize: 150 180 74"
    print("maxsize:", maxsize1, maxsize2, maxsize3)

def input_visual():
    # 预测结果保存根路径
    test_root_path = "../../input/"
    # 测试结果保存路径
    test_save_path = os.path.join(test_root_path, "isles2022", "test") + "/"

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    db_test = ISLES2022(base_dir='../../dataset',
                        split='test',
                        transform_1=get_test_transform(),
                        )

    for i in range(len(db_test)):

        ids = i + 201

        print("--------第{}个样本--------".format(i))
        img = db_test[i]['image'].as_tensor()
        label = db_test[i]['label'].as_tensor()

        print("img", img.shape)
        print("label", label.shape)
        print("img.max()", img.max())
        print("img.min()", img.min())
        print("img.sum()", img.sum())
        # print("label.max()", label.max())
        # print("label.min()", label.min())
        # print("label.sum()", label.sum())

        img = img[0, ...].cpu().data.numpy()
        label = label[0, ...].cpu().data.numpy()

        img_itk = sitk.GetImageFromArray(img)
        img_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img_itk, test_save_path +
                        "Sample_{}_img.nii.gz".format(ids))

        lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
        lab_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(lab_itk, test_save_path +
                        "Sample_{}__lab.nii.gz".format(ids))

if __name__ == "__main__":
    size_range()










