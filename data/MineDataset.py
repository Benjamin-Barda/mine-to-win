from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class MineDataset(Dataset):
    
    def __init__(self, img_dir, JSON_dir, transform = None):
        """_summary_

        Args:
            img_dir (string): Path to the directory containing all the dataset's images
            JSON_dir (string): Path to the JSON containing every image's respective label
            transform (callable, optional): An optional transform to apply on a sample. Defaults to None.
        """
        self.data = pd.read_json(JSON_dir).T.drop(["written"], axis=1)
        
        self.data = self.data.drop(self.data[self.data.purge].index).drop(["purge"], axis=1)
        
        self.images = torch.Tensor(np.array(
            [cv2.cvtColor(cv2.imread(os.path.join(img_dir, self.data.iloc[i].name)), cv2.COLOR_BGR2RGB) 
             for i in range(len(self.data))
             ]))
        
        self.data["Image"] = [img for img in self.images]
        self.JSON_DIR = JSON_dir
        self.transform = transform
        
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, indx):

        row = self.data.iloc[indx]
        
        if self.transform:
            row["Image"] = self.transform(row["Image"])
            
        return row 

class MineDatasetMulti(Dataset):
    
    def __init__(self, load_dir, name, transform = None):
        """_summary_

        Args:
            img_dir (string): Path to the directory containing all the dataset's images
            JSON_dir (string): Path to the JSON containing every image's respective label
            transform (callable, optional): An optional transform to apply on a sample. Defaults to None.
        """
        self.transform = transform
        self.data, self.JSON_DIR, self.img_dir = torch.load(load_dir + "\\" + name + ".dtset")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if not np.isscalar(idx):
            images = torch.FloatTensor([cv2.imread(self.img_dir + "\\" + name) for name in  self.data.iloc[idx].index]).permute(0,3,1,2) / 255
            if self.transform:
                images = self.transform(images)
            infos = self.data.iloc[idx]
                
            return images, infos
        
        image = torch.FloatTensor(cv2.imread(os.path.join(self.img_dir, self.data.iloc[idx].name))).permute(2,0,1) / 255
        if self.transform:
            image = self.transform(image)
        info = self.data.iloc[idx]
            
        return image, info

def create_dataset(JSON_dir, img_dir, path, name):
    data = pd.read_json(JSON_dir[0]).T
    data.drop(["written"], axis=1, inplace=True)
    data.drop(data[data.purge].index, inplace=True)
    data.drop(["purge"], axis=1, inplace=True)
    # self.images = torch.ByteTensor(np.array(
    #         [cv2.cvtColor(cv2.imread(os.path.join(img_dir[0], self.data.iloc[i].name)), cv2.COLOR_BGR2RGB) 
    #         for i in range(len(self.data))
    #         ]))

    for filename in JSON_dir[1:]:
        df = pd.read_json(filename).T
        df.drop(["written"], axis=1, inplace=True)
        df.drop(df[df.purge].index, inplace=True)
        df.drop(["purge"], axis=1, inplace=True)
        # self.images = torch.cat((self.images,torch.ByteTensor(np.array(
        #     [cv2.cvtColor(cv2.imread(os.path.join(img, df.iloc[i].name)), cv2.COLOR_BGR2RGB) 
        #     for i in range(len(df))
        #     ]))), 0)
        data = pd.concat([data, df], copy=False)

    torch.save((data, JSON_dir, img_dir), path + "\\" + name + ".dtset")
    #torch.save(self.images.to(dtype=torch.float32) / 255, path + name + "_images.dtset")