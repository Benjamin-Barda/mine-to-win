import cv2
import os
import random
import numpy as np
import torch
from torchvision import transforms

transforms = transforms.Compose((
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05), saturation=(0.99, 1.01)),
    transforms.RandomAdjustSharpness(1.25, p=0.2),
    transforms.RandomAdjustSharpness(0.75, p=0.2),
    transforms.RandomRotation((-15, 15)),
    ))

random.seed(42)

files = os.listdir("./imgs/frames_new")

indexes = random.choices(list(range(0, len(files))), k=144)

img = np.zeros((720,720,3), dtype=np.uint8)

t = 0
for y in range(16):
    for x in range(9):
        curr_img = cv2.imread("./imgs/frames_new/" + files[indexes[t]])
        mod_img = transforms(curr_img)
        fin_img = mod_img.numpy()
        fin_img = fin_img.transpose(1,2,0)
        frame = cv2.resize(fin_img * 255,(80,45), interpolation=cv2.INTER_AREA)
        img[y*45:(y+1)*45, x*80:(x+1)*80] = frame
        t += 1

cv2.imshow("Final img mod", img)
cv2.waitKey(0)

cv2.imwrite("./docs/images/dtset_repr_mod.png", img)
