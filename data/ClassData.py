
import torch
import pandas as pd
from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np

class ClassData(data.Dataset):

    def __init__(self, sources = [],  transform = None, transform_target = None, device = "cpu", train = True):

        self.transform = transform
        self.transform_target = transform_target
        self.train = train
        df_iter = (pd.read_json(src).T for src in sources)
        df = pd.concat(df_iter).T.drop(["purge", "written"]).T.sample(frac=1)
        self.images = np.asarray(
                [cv2.imread("imgs\\frames_new\\" + i) 
                for i in df.index]
                )

        self.images = np.asarray(self.images, dtype = "uint8")
        self.labels = np.asarray( 4 * df["Sheep"].astype(bool).astype(int) + 3 * df["Zombie"].astype(bool).astype(int)
                                + 2 * df["Pig"].astype(bool).astype(int) + df["Creeper"].astype(bool).astype(int), dtype=np.ubyte)

        boundings = list(zip(df["Creeper"], df["Pig"], df["Zombie"], df["Sheep"]))
        
        vertices = np.empty((0), dtype = np.float32)
        elements = np.empty((len(boundings), 2), dtype = np.int32)
        vertices_l = np.empty(0, dtype = np.ubyte)

        previous = 0

        for i, vertex in enumerate(boundings):
            
            elements[i][0] = previous
            
            for label, mobbox in enumerate(vertex):
                
                vertices = np.hstack((vertices, np.asarray(mobbox).flatten()))
                previous += 4 * len(mobbox)
                
                vertices_l = np.hstack((vertices_l, np.asarray([label+1]  * len(mobbox), dtype=np.ubyte)))
                
            elements[i][1] = previous
    
        self.vertices   = torch.tensor(vertices, dtype = torch.float32)
        self.elements   = torch.tensor(elements, dtype = torch.int32)
        self.vertices_l = torch.tensor(elements, dtype = torch.uint8)
        
        assert(len(self.images) == len(self.labels) == len(boundings))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()

        img = self.images[index]
        if self.train:
            if self.transform:
                img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = self.labels[index]

        if self.train and self.transform_target:
            label = self.transform_target(label)
        
        elements = self.elements[index]

        return img, label, elements

    def shape(self):
        return self.images.shape
    
    def getvertex(self, elements):
        if torch.is_tensor(elements):
            elements = elements.tolist()
        
        base, top = elements
        return self.vertices[base:top].reshape(-1, 4), self.vertices_l[base>>2:top>>2].reshape(-1, 1)
    
    def tocuda(self):
        
        self.vertices_l.cuda(non_blocking=True)
        self.elements.cuda(non_blocking=True)
    
    def set_train_mode(self, is_training):
        self.train = is_training
    
def create_and_save_dataset(srcs, filename):
    from torchvision import transforms
    
    trans = transforms.Compose((
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05), saturation=(0.99, 1.01)),
    transforms.RandomAdjustSharpness(1.25, p=0.2),
    transforms.RandomAdjustSharpness(0.75, p=0.2),
    #transforms.RandomRotation((-15, 15)),
    ))

    ds = ClassData(srcs, transform=trans)
    
    torch.save(ds, filename)

def create_and_save_dataset_test(srcs, filename):
    from torchvision import transforms
    
    trans = transforms.Compose((
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05), saturation=(0.99, 1.01)),
    # transforms.RandomAdjustSharpness(1.25, p=0.2),
    # transforms.RandomAdjustSharpness(0.75, p=0.2),
    #transforms.RandomRotation((-15, 15)),
    ))

    ds = ClassData(srcs, transform=trans)
    
    torch.save(ds, filename)

if __name__ == "__main__":
    # names = [f"jsons\\rePARSEDOUTPUT-creeper{i}.json" for i in range(1, 11)]
    # names += [f"jsons\\rePARSEDOUTPUT-pig{i}.json" for i in range(1, 11)]
    # #names += [f"jsons\\rePARSEDOUTPUT-null{i}.json" for i in range(1, 20)]
    # names += [f"jsons\\rePARSEDOUTPUT-zombie{i}.json" for i in range(1, 10)]
    # names += [f"jsons\\rePARSEDOUTPUT-sheep{i}.json" for i in range(1, 10)]
    # #names.append("jsons\\rePARSEDOUTPUT-seanull1.json")
    # create_and_save_dataset(names, "data\\datasets\\minedata_compressed_local.dtst")

    names = [f"jsons\\PARSEDOUTPUT-test{i}.json" for i in range(1, 3)]
    create_and_save_dataset_test(names, "data\\datasets\\minedata_compressed_local_test.dtst")

    # names = [f"jsons\\PARSEDOUTPUT-creeper{i}.json" for i in range(1, 11)]
    # names += [f"jsons\\PARSEDOUTPUT-pig{i}.json" for i in range(1, 11)]
    # names += [f"jsons\\PARSEDOUTPUT-null{i}.json" for i in range(1, 20)]
    # names += [f"jsons\\PARSEDOUTPUT-zombie{i}.json" for i in range(1, 10)]
    # names += [f"jsons\\PARSEDOUTPUT-sheep{i}.json" for i in range(1, 10)]
    # names.append("jsons\\PARSEDOUTPUT-seanull1.json")
    # create_and_save_dataset(names, "data\\datasets\\minedata_classifier_colab.dtst")
