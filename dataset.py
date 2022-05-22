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


class Selfie_2_Anime(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        f_name = self.data_list[i]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transf_img = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        img = Image.open(f_name)
        img = transf_img(img)

        return img

