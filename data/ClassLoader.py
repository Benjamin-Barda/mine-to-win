
import torch
import pandas as pd
from torch.utils import data
import cv2
import numpy as np

class ClassData(data.Dataset):

    def __init__(self, sources = [],  transform = None, device = "cpu"):


        self.transform = transform
        df_iter = (pd.read_json(src).T for src in sources)
        df = pd.concat(df_iter).T.drop(["purge", "written"]).T.sample(frac=1)
        self.images = np.asarray(
                [cv2.imread("imgs\\frames\\" + i) 
                for i in df.index]
                )

        self.images = torch.ByteTensor(self.images).permute(0,3,1,2)
        self.labels = torch.ByteTensor( 2 * df["Pig"].astype(bool).astype(int) 
                                          + df["Creeper"].astype(bool).astype(int) )

        self.boundings = list(zip(df["Pig"], df["Creeper"]))

        assert(len(self.images) == len(self.labels) == len(self.boundings))

        if device == "cuda":
            self.images.pin_memory()
            self.labels.pin_memory()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
                    index = index.tolist()

        img = self.images[index]
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[index]

        sample = (img, label)
        return sample

    def shape(self):
        return self.images.shape