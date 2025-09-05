import os
from typing import Dict

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
from monai import transforms

patch_size = (112, 112, 80)

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train',transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/LA_2018/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split=='test':
            with open(self._base_dir + '/LA_2018/test.list', 'r') as f:
                self.image_list = f.readlines()
        else:
            self.image_list = None

        self.image_list = [item.replace('\n','') for item in self.image_list]   #获取名称列表

        # print("image_list:",self.image_list)
        # print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/LA_2018/"+image_name+"/mri_norm2.h5", 'r')   #打开h5文件
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}   #获取图像标签字典

        # data_path =self._base_dir+"/LA_2018/"+image_name+"/mri_norm2.h5"
        # img_nib_path = os.path.join(data_path, 'dwi_case.nii')
        # label_nib_path = os.path.join(data_path, 'label_case.nii')
        # sample = {'image': img_nib_path, 'label': label_nib_path}  # 获取图像标签字典
        # print("sample:", sample)

        "可以利用不同transform对数据进行不同处理"
        if self.transform:
            sample = self.transform(sample)

        return sample




def get_train_transform():
    # 定义转换
    train_transform = [
        Normalize(),
        RandomCrop(patch_size),
        RandomRotFlip(),
        ToTensor()
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform

def get_test_transform():
    # 定义转换
    test_transform = [
        Normalize(),
        ToTensor()
    ]
    test_transform = transforms.Compose(test_transform)
    return test_transform

class Normalize(object):
    """ 标准化图像强度（根据医学图像特性调整） """
    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        image = (image - image.min()) / (image.max() - image.min())  # [0, 1]
        return {'image': image, 'label': label}

class ToTensor(object):
    """ 将numpy数组转为torch张量，并处理医学图像的通道维度 """
    def __call__(self, sample: Dict) -> Dict:
        image, label = sample['image'], sample['label']
        # 添加通道维度 (C, D, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)  # 1×D×H×W
        label = torch.from_numpy(label).long().unsqueeze(0)   # 1×D×H×W
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}



class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image = sample['image']
#         image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



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


if __name__ == "__main__":
    db_train = LAHeart(base_dir='../../dataset',
                       split='train',
                         # todo change training flod
                       transform=get_train_transform_1())
    img = db_train[0]['image']
    label = db_train[0]['label']

    print("img", img.shape)
    print("label", label.shape)
    print("img.max()", img.max())
    print("img.min()", img.min())
    print("img.sum()", img.sum())
    print("img.avg:", img.sum()/img.numel())
    print("label.sum()", label.sum())
