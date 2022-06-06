import sys
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.loss_func import RPNLoss
from models.regionProposal.utils.anchorUtils import *
from torch.utils.data import DataLoader, Subset
import cv2 as cv
import torch
import numpy as np
from data.ClassData import ClassData

DEBUG = False
SHOW = True

bs = 1

# Loading only one image
ds = torch.load("data\\datasets\\minedata_compressed_local_test.dtst")
dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=True)

state_extractor, state_rpn = torch.load("./weights_too_heavy/MineRPN_quantized_weights.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=False).to(("cpu"))
extractor.load_state_dict(state_extractor)
extractor.eval()
extractor = torch.jit.freeze(torch.jit.script(extractor))

lossfn = RPNLoss()

_, inDim, hh, ww, = extractor(ds[0][0][None, ...]).size()

cv.namedWindow("fmap1", cv.WINDOW_NORMAL)
cv.namedWindow("fmap2", cv.WINDOW_NORMAL)
cv.namedWindow("fmap3", cv.WINDOW_NORMAL)
cv.namedWindow("fmap4", cv.WINDOW_NORMAL)
cv.namedWindow("fmap5", cv.WINDOW_NORMAL)
cv.namedWindow("o_img", cv.WINDOW_NORMAL)


with torch.no_grad():
    
    for i in range(200, 0, -1):

        print(i)

        img, lbl, elem = ds[i]
        img = img[None, ...]
        base_feat_map = extractor(img)

        fmaps = base_feat_map.permute(0, 2, 3, 1)[0, ...].numpy()
        img = img.permute(0, 2, 3, 1)[0, ...].numpy()

        cv.imshow("fmap1", fmaps[:,:,0])
        cv.imshow("fmap2", fmaps[:,:,1])
        cv.imshow("fmap3", fmaps[:,:,2])
        cv.imshow("fmap4", fmaps[:,:,3])
        cv.imshow("fmap5", fmaps[:,:,4])
        cv.imshow("o_img", img)
        cv.waitKey(0)