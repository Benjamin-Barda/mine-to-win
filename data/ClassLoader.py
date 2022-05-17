
import torch
import pandas as pd
from torch.utils import data
import cv2
import numpy as np

class ClassLoader(data.Dataset):


    def __init__(self, sources = [],  transform = None):

        self.transform = transform
        df_iter = (pd.read_json(src).T for src in sources)
        df = pd.concat(df_iter).T.drop(["purge", "written"]).T
        print(df)
        self.images = np.asarray(
                [cv2.imread("imgs\\frames\\" + i) 
                for i in df.index]
                )

        self.images = torch.ByteTensor(self.images)

        self.labels = torch.ByteTensor( 2 * df["Pig"].astype(bool).astype(int) 
                                          + df["Creeper"].astype(bool).astype(int) )

        self.boundings = list(zip(df["Pig"], df["Creeper"]))

        assert(len(self.images) == len(self.labels) == len(self.boundings))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[index]

        return img, label


            