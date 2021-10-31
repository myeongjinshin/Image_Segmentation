import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import urllib
import glob
import skimage.io as skio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class Relabel:
    def __init__(self, inlabel, outlabel):
        self.inlabel = inlabel
        self.outlabel = outlabel

    def __call__(self, tensor):
        tensor[tensor == self.inlabel] = self.outlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(2):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:
    def __init__(self, n=9):
        self.cmap = colormap(10)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])


    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class CoE_Dataset(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, crop_size=192, is_train=True, array=[]):
        self.images_root = os.path.join(root, 'Images')
        self.labels_root = os.path.join(root, 'Labels')
        self.array = array
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        
        self.is_train = is_train
        if self.is_train:
            self.length = 1800
        else:
            self.length = 100

        self.filenames.sort()
        #random.shuffle(self.filenames)

        self.crop_size = crop_size
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.noise = torch.randn(self.length, 3, crop_size, crop_size)

    def __getitem__(self, index):
        if self.is_train:
            filename = self.filenames[self.array[index%900]]
            with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
                image = load_image(f).convert('RGB')
            with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
                label = load_image(f).convert('P')

            #transforms.RandomRotation(10)
            #transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2))

            if random.random() > 0.8:
                brightness_factor, contrast_factor, saturation_factor, hue_factor = random.uniform(0.9, 1.0), random.uniform(0.8, 0.9), random.uniform(0.7, 0.9), random.uniform(0, 0.2)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
                image = TF.adjust_saturation(image, saturation_factor)
                image = TF.adjust_hue(image, hue_factor)

            if random.random() > 0.7:
                affine_params = transforms.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), (self.crop_size, self.crop_size))
                image, label = TF.affine(image, *affine_params), TF.affine(label, *affine_params)

            if min(image.size[0],image.size[1])<self.crop_size:
                image = transforms.CenterCrop(self.crop_size)(image)
                label = transforms.CenterCrop(self.crop_size)(label)

            else:
                i, j, h, w = transforms.RandomCrop.get_params(image,(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                label = TF.crop(label, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

            if self.input_transform is not None:
                image = self.input_transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)

            #image = image + 0*self.noise[index]
        else:
            filename = self.filenames[self.array[index]]
            with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
                image = load_image(f).convert('RGB')
            with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
                label = load_image(f).convert('P')

            #image = transforms.CenterCrop(self.crop_size)(image)
            #label = transforms.CenterCrop(self.crop_size)(label)

            if self.input_transform is not None:
                image = self.input_transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)

        return image, label, filename

    def __len__(self):
        return self.length


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    epsilon = 1e-12
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    
    return area_inter.float(), area_union.float() + epsilon

