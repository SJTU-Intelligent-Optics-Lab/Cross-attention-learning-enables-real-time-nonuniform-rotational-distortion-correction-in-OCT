# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import math
import os
import cv2
import numpy as np
import torch

from monai import data, transforms

from monai.data import load_decathlon_datalist

import PIL
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    if args.test_mode:
        dataset_test = build_dataset_and_transform(args.val_text, args=args)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_test, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False)
        loader = data_loader_val

    else:
        dataset_train = build_dataset_and_transform(args.train_text, args=args)
        dataset_val = build_dataset_and_transform(args.val_text, args=args)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle = True,
            drop_last=False)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False)
        loader = [data_loader_train, data_loader_val]

    return loader

def build_dataset_and_transform(split, args):
    transform = build_transform(split, args)

    dataset = build_dataset(args.data_dir, args.list_dir, split, transform=transform)
    if 'train' in split:
        print(len(dataset),"train data total")
    if 'val' in split:
        print(len(dataset), "val data total")

    return dataset

def build_transform(split, args):
    if 'train' in split:
        transform_image = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Normalize(mean=[0], std=[1]),
            #transforms.Normalize(mean=[0.227,0.227,0.227], std=[0.1935,0.1935,0.1935]),
            transforms.ToTensor(),
        ])
        return transform_image

    elif 'val' in split:
        transform_image = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Normalize(mean=[0], std=[1]),
            #transforms.Normalize(mean=[0.227,0.227,0.227], std=[0.1935,0.1935,0.1935]),
            transforms.ToTensor(),
        ])
        return transform_image

class build_dataset(Dataset):
    def __init__(self, data_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = data_dir
        f = open(os.path.join(list_dir, self.split+'.txt'))
        txt_list = f.readlines()
        self.lst = []
        for i in range(len(txt_list)):
            line = txt_list[i].strip().split(' ')
            info = []
            info.append(line[0])
            info.append(np.expand_dims(np.array(list(map(float, line[1:]))), axis=-1))
            self.lst.append(info)
        f.close()

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        slice_name = self.lst[idx][0]
        label = self.lst[idx][1]
        # labelNoise=np.random.randint(-4,5,size=(1024,1))
        # label=label+labelNoise

        #image_path = os.path.join(self.data_dir, self.split, slice_name)
        if 'train' in self.split:
            obj_path = os.path.join(self.data_dir, 'data augmentation_distortion', slice_name)
            ref_path = os.path.join(self.data_dir, 'data augmentation_reference', slice_name)
        elif 'val' in self.split:
            obj_path = os.path.join(self.data_dir, 'data augmentation_distortion', slice_name)
            ref_path = os.path.join(self.data_dir, 'data augmentation_reference', slice_name)
        else:
            raise 'can not find the train or val dict'
        ref = cv2.imread(ref_path,cv2.IMREAD_GRAYSCALE)
        ref = np.transpose(ref,(1, 0))
        # 给图片添加speckle噪声
        gauss = np.random.randn(1024,512)/8
        ref = ref + ref * gauss
        # 归一化图像的像素值
        ref = np.clip(ref, a_min=0, a_max=255)
        ref = np.round(ref).astype(np.uint8)

        obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
        obj = np.transpose(obj, (1, 0))
        # 给图片添加speckle噪声
        gauss = np.random.randn(1024,512)/8
        obj = obj + obj * gauss
        # 归一化图像的像素值
        obj = np.clip(obj, a_min=0, a_max=255)
        obj = np.round(obj).astype(np.uint8)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            ref = self.transform(ref)
            torch.random.manual_seed(seed)
            obj = self.transform(obj)

        return [ref, obj, label, slice_name]





class RandomResize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1], size=(224,224)):
        self.scales = scales
        self.size = size
    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            tmp = sample[elem]
            if elem =="label":
                flagval = cv2.INTER_NEAREST
            elif elem =="image":
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)
            tmp = cv2.resize(tmp, self.size,interpolation=flagval)
            sample[elem] = tmp

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if elem == 'image':
                tmp = sample[elem]
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            elif elem == 'label':
                tmp = sample[elem]
                tmp = tmp[:,:,np.newaxis]
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample

class RandomFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):
        # Horizontal flipping
        if random.random() < 0.5:
            for elem in sample.keys():

                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        # Vertical flipping
        if random.random() < 0.5:
            for elem in sample.keys():

                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=0)
                sample[elem] = tmp

        return sample

class Normalize(object):
    def __init__(self, mean, std, inplace=False):

        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        tmp = sample['image'].float()
        tmp =  F.normalize(tmp, self.mean, self.std, self.inplace)
        sample['image'] = tmp

        return sample