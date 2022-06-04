import sys
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.loss_func import RPNLoss
from models.regionProposal.utils.anchorUtils import *
from torch.utils.data import DataLoader, Subset
import cv2 as cv
import torch
from data.ClassData import ClassData

DEBUG = False
SHOW = True

bs = 1

# Loading only one image
ds = torch.load("data\\datasets\\minedata_compressed_local.dtst")
dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=True)

sum_w = 0
sum_h = 0
n = 0

for _, _, elements in iter(dl):
    bounds, _ = ds.getvertex(elements[0])
    for box in enumerate(bounds[0]):
        _,_,w,h = box[1]
        sum_w += w
        sum_h += h
        n += 1

print(sum_w / n, sum_h / n)