from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class BackboneDataset(Dataset):
    
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