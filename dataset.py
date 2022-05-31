import torch
import PIL
from torch.utils.data import Dataset
import os
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

def get_full_list(root_dir,sub_dir):
    data_list = []

    data_dir = osp.join(
        root_dir, sub_dir
    )

    data_list += sorted(
        osp.join(data_dir, img_name) for img_name in
        filter(
            lambda x: x[-4:] == '.jpg',
            os.listdir(data_dir)
        )
    )
    return data_list


class ImageDataset(Dataset):
    def __init__(self, data_list, ):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        f_name = self.data_list[i]

        # normalize pixel values to [-1., 1] range
        transf_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        img = Image.open(f_name)
        img = transf_img(img)

        return img
        
    
class Selfie2Anime_Dataset(Dataset):
    '''
    combine selfie dataset and anime dataset together
    
    return (selfie_image, anime_image)
    
    '''
    
    def __init__(self, selfie_dataset, anime_dataset):
        if len(selfie_dataset) != len(anime_dataset):
            raise AttributeError("Dataset size does not match!")
            
        self.selfie_dataset = selfie_dataset
        self.anime_dataset = anime_dataset
        
    def __len__(self):
        return len(self.selfie_dataset)

    def __getitem__(self, i):
        selfie_img = self.selfie_dataset[i]
        anime_img = self.anime_dataset[i]

        return (selfie_img, anime_img)

