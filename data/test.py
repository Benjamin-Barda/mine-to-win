from ClassLoader import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import cv2

transforms = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05))
)

ds = ClassData(["jsons\\OUTPUT-creeper1.json"], transform=transforms)
ds_load = DataLoader(ds, batch_size=16, shuffle=True)

for _ in range(20):
    for img, label in ds_load:
        
        cv2.imshow("Sample", img.permute(0,2,3,1)[0].numpy())
        cv2.waitKey(-1)
        break

cv2.destroyAllWindows()