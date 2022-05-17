from ClassLoader import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
import cv2

transforms = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05))
)

ds = ClassData(["jsons\\OUTPUT-creeper1.json"], transform=transforms)



split = int(ds.shape()[0] * 0.8)
train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, ds.shape()[0])))

train_load = DataLoader(train, batch_size=32, shuffle=True)
val_load =   DataLoader(val, batch_size=32, shuffle=True)


for _ in range(20):
    for img, label in train_load:
        
        cv2.imshow("Sample", img.permute(0,2,3,1)[0].numpy())
        cv2.waitKey(-1)

    print("train done")
    for img, label in val_load:
        
        cv2.imshow("Sample", img.permute(0,2,3,1)[0].numpy())
        cv2.waitKey(-1)

    print("valid done")


cv2.destroyAllWindows()
