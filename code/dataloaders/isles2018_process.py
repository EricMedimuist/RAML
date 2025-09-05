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


roi_size=(128, 128, 32)
test_size=(128, 128, 32)

class ISLES2018(Dataset):
    """ ISLES Dataset """
    def __init__(self, base_dir=None, split='train',train_samples_num=75, transform_1=None):
        self._base_dir = base_dir
        self.transform_1= transform_1
        self.sample_list = []
        self.name_list = []
        self.train_samples_num = train_samples_num
        self.test_samples_num = 19
        self.split=split


        if split == 'train':
            for i in range(self.test_samples_num+1,train_samples_num+self.test_samples_num+1):
                self.name_list.append(str(i))
        elif  split == 'test':
            for i in range(1,self.test_samples_num+1):
                self.name_list.append(str(i))

        # print("image_list:",self.name_list)
        # print("total {} samples".format(len(self.name_list)))

    def __len__(self):
        return  len(self.name_list)

    def __getitem__(self, idx):
        "根据idx索引读取文件"

        data_path=self._base_dir+"/ISLES_2018/Sample_"+ self.name_list[idx]
        img_nib_path = os.path.join(data_path,'dwi_case.nii')
        label_nib_path =os.path.join(data_path, 'label_case.nii')

        sample = {'image': img_nib_path, 'label': label_nib_path }   #获取图像标签字典
        # print("sample",sample)


        if self.transform_1:
            sample = self.transform_1(sample)


        return   sample

def get_train_transform_1():
    # 定义转换
    train_transform = [
        transforms.LoadImaged(keys=["image", "label"]),  # 加载图像和标签
        transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 确保通道维度在第 1 维
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear")),  # 统一体素间距
        transforms.CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),  # 裁剪前景区域


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
        transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),  # 随机旋转 90 度
        transforms.RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),  # 随机缩放

        # transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # 强度归一化到 [0, 1]
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
        transforms.CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),  # 裁剪前景区域


        transforms.Resized(["image", "label"], spatial_size=test_size, size_mode='all', mode="bilinear",align_corners=False),
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

        transforms.Resized(["image", "label"], spatial_size=test_size, size_mode='all', mode="bilinear",align_corners=False),

        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),  # 确保数据为 Tensor 格式

    ]
    test_transform = transforms.Compose(test_transform)
    return test_transform



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


def convert():
    import os
    root_path = '../../dataset'

    for i in range(1, 95):
        new_folder = os.path.join(root_path, 'ISLES_2018/Sample_' + str(i))
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)

        os.makedirs(new_folder)

    # 指定要搜索的关键字
    keyword_1 = "DWI"
    keyword_2 = "OT"

    # 遍历目录并查找文件名中包含关键字的文件
    matching_files = []
    for i in range(1, 95):
        directory = os.path.join(root_path, "ISLES_2018_raw", "case_" + str(i))
        # print(" directory:", directory)
        for root, dirs, files in os.walk(directory):  # 遍历文件夹
            # print("root:",root)
            # print("dirs:", dirs)
            # print("files:",files)

            for file in files:
                if keyword_1 in file:  # 检查文件名是否包含关键字
                    # matching_files.append(os.path.join(root, file))  # 保存完整路径
                    img_source = os.path.join(root, file)
                    print(" img_source:", img_source)

                    img_destiny = os.path.join(root_path, 'ISLES_2018', 'Sample_' + str(i), 'dwi_case.nii')
                    print(" img_destiny:", img_destiny)

                    shutil.copy(img_source, img_destiny)

                if keyword_2 in file:  # 检查文件名是否包含关键字
                    # matching_files.append(os.path.join(root, file))  # 保存完整路径
                    lab_source = os.path.join(root, file)
                    print(" lab_source :", lab_source)

                    lab_destiny = os.path.join(root_path, 'ISLES_2018', 'Sample_' + str(i), 'label_case.nii')
                    print(" lab_destiny :", lab_destiny)

                    shutil.copy(lab_source, lab_destiny)

def convert1():
    from scipy.ndimage import binary_erosion, binary_dilation

    root_path = "../../dataset/"
    ori_dataset_name = "ISLES_2018_ori"
    new_dataset_name = "ISLES_2018"

    for i in range(1, 95):
        data_load_path = os.path.join(root_path, ori_dataset_name,"Sample_"+str(i))
        data_save_path = os.path.join(root_path, new_dataset_name,"Sample_"+str(i))

        "创建样本路径"
        if os.path.exists(data_save_path):
            shutil.rmtree(data_save_path)
        os.makedirs(data_save_path)

        nii_img = nib.load(data_load_path+"/dwi_case.nii")  # 加载 NIfTI 文件
        nii_label = nib.load(data_load_path + "/label_case.nii")  # 加载 NIfTI 文件

        data = nii_img.get_fdata()  # 获取图像数据为 numpy 数组
        affine = nii_img.affine  # 获取仿射变换矩阵
        header = nii_img.header  # 获取头信息

        data_l = nii_label.get_fdata()  # 获取图像数据为 numpy 数组
        affine_l = nii_label.affine  # 获取仿射变换矩阵
        header_l = nii_label.header  # 获取头信息



        # brain_mask= data > 50
        # brain_mask = binary_dilation(brain_mask, iterations=2)  # 膨胀
        # brain_mask = binary_erosion(brain_mask, iterations=2)  # 腐蚀


        data[data < 50] = 0

        new_nii_img = nib.Nifti1Image(data, affine, header)
        nib.save(new_nii_img, data_save_path+"/dwi_case.nii")

        new_nii_label = nib.Nifti1Image(data_l, affine_l, header_l)
        nib.save(new_nii_label, data_save_path + "/label_case.nii")

        print("--------sample_{} is done----------".format(i))

def skull_strip_fsl():
    import subprocess
    """
    使用 FSL 的 BET 工具对脑图像进行颅骨剥离
    Args:
        input_file (str): 输入的 NIfTI 图像路径 (.nii 或 .nii.gz)
        output_file (str): 输出的颅骨剥离后的 NIfTI 图像路径
    """
    root_path = "../../dataset/"
    input_file = root_path+"ISLES_2018_ori/Sample_1"  # 输入的 DWI 图像
    output_file = root_path+"ISLES_2018_new/Sample_1"  # 输出的颅骨剥离后的图像

    if os.path.exists(output_file):
        shutil.rmtree(output_file)
    os.makedirs(output_file)

    try:
        # 调用 FSL 的 BET 工具
        subprocess.run(["bet", input_file+"/dwi_case.nii", output_file+"/dwi_case.nii", "-f", "0.5", "-g", "0"], check=True)
        print(f"颅骨剥离完成，结果已保存到: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"运行 FSL 的 BET 工具失败: {e}")

def skull_strip_ants(input_file, output_file, template_file, template_mask):
    import ants
    """
    使用 ANTs 对脑图像进行颅骨剥离
    Args:
        input_file (str): 输入的 NIfTI 图像路径 (.nii 或 .nii.gz)
        output_file (str): 输出的颅骨剥离后的 NIfTI 图像路径
        template_file (str): 标准模板（如 MNI 模板）的路径
        template_mask (str): 标准模板的脑掩膜路径
    """
    # 加载输入图像、模板和模板掩膜
    img = ants.image_read(input_file)
    template = ants.image_read(template_file)
    mask = ants.image_read(template_mask)

    # 进行脑提取（颅骨剥离）
    brain = ants.brain_extraction(img, template, mask)

    # 保存结果
    ants.image_write(brain, output_file)
    print(f"颅骨剥离完成，结果已保存到: {output_file}")



def size_range():
    db_train = ISLES2018(base_dir='../../dataset',
                         split='train',
                         transform_1=get_train_transform_1(),
                         )

    # print("db_train", db_train[19])
    minsize1 = 1000
    minsize2 = 1000
    minsize3 = 1000

    maxsize1 = 0
    maxsize2 = 0
    maxsize3 = 0
    for i in range(len(db_train)):
        print("size of sample_{}".format(i))
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

    "foreground minsize: 60 71 52"
    print("minsize:", minsize1, minsize2, minsize3)
    "foreground maxsize: 76 92 74"
    print("maxsize:", maxsize1, maxsize2, maxsize3)
# 定义保存 Tensor 为 NIfTI 文件的函数
def save_tensor_as_nii(tensor, output_path, affine=None):
    """
    保存 PyTorch Tensor 为 NIfTI 文件
    Args:
        tensor (torch.Tensor): 要保存的 3D 或 4D Tensor
        output_path (str): 保存文件路径（包括 .nii 或 .nii.gz）
        affine (np.ndarray): 仿射矩阵，默认为单位矩阵
    """
    # 检查 Tensor 是否在 GPU 上，如果是，移到 CPU
    # if tensor.is_cuda:
    #     tensor = tensor.cpu()

    # 转换为 NumPy 数组
    # np_data = tensor.numpy()

    # 如果没有提供仿射矩阵，使用单位矩阵
    if affine is None:
        affine = np.eye(4)

    # 创建 NIfTI 对象
    nii_image = nib.Nifti1Image(tensor, affine)

    # 保存为 NIfTI 文件
    nib.save(nii_image, output_path)
    print(f"Tensor 已保存为 NIfTI 文件: {output_path}")

def input_visualization():


    "保存路径"
    test_root_path = "../../input/"
    dataset_name = "ISLES2018"
    exp_name = "train"
    test_save_path = os.path.join(test_root_path, dataset_name, exp_name) + "/"

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    db_train = ISLES2018(base_dir='../../dataset',
                         split=exp_name,
                         transform_1=get_test_transform(),
                         )

    k = len(db_train)
    for i in range(k):
        id = i + 1
        print("--------第{}个样本--------".format(id))
        "[1, 1, 64, 64, 64]"
        image = db_train[i]['image'].as_tensor().unsqueeze(dim=0)
        label = db_train[i]['label'].as_tensor().unsqueeze(dim=0)
        image, label = image.cuda(), label.cuda()
        # print("image.shape:",image.shape)
        # print("label.shape:",label.shape)

        "[64, 64, 64]"
        image = image[0, 0, ...].cpu().data.numpy()
        # print(" image.shape:", image.shape)

        "[64, 64, 64]"
        label = label[0, 0, ...].cpu().data.numpy()
        # print(" label.shape:", label.shape)

        # 创建 sitk 图像对象
        # img_itk = sitk.GetImageFromArray(image)
        # img_itk.SetSpacing((1.0, 1.0, 1.0))
        # sitk.WriteImage(img_itk, test_save_path +
        #                 "Sample_{}_img.nii".format(id))
        #
        # lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
        # lab_itk.SetSpacing((1.0, 1.0, 1.0))
        # sitk.WriteImage(lab_itk, test_save_path +
        #                 "Sample_{}_lab.nii".format(id))

        #  nibal 保存tensor
        save_tensor_as_nii(image, test_save_path +"Sample_{}_img.nii".format(id))
        save_tensor_as_nii(label, test_save_path + "Sample_{}_lab.nii".format(id))



if __name__ == "__main__":
    convert1()









