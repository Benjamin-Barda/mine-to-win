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

state_extractor, state_rpn = torch.load("./MineRPN_best_weights.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(state_extractor)
lossfn = RPNLoss()

cv.namedWindow("img", cv.WINDOW_NORMAL)

with torch.no_grad():
    
    
    for i in range(100, 103):
        extractor.eval()

        img, lbl, bounds = ds[i]
        img = img[None, ...]
        base_feat_map = extractor.forward(img)
        _, inDim, hh, ww, = base_feat_map.size()

        
        rpn = _rpn(inDim).to(("cpu"))
        rpn.load_state_dict(state_rpn)
        rpn.eval()

        score, reg, rois = rpn(base_feat_map, img.shape[-2:])
        
        print(torch.min(rois[:, :, 1:4:2]), torch.max(rois[:, :, 0:4:2]), torch.mean(rois[:, :, 0:4:2]), torch.std(rois[:, :, 0:4:2]))
        img = img.permute(0, 2, 3, 1)[0, ...].numpy()
        img = np.ascontiguousarray(img)
        cv.imshow("orig img", img)

        for indx, an in enumerate(rois[0]):
            col = (0,255,0)
            if score[0][indx] < .5:
                col = (0,0,255)
            x,y,w,h = an.int()

            x1,y1,x2,y2 = int(x.item() - w.item()//2), int(y.item() - h.item()//2), int(x.item() + w.item()//2), int(y.item() + h.item()//2)
            cv.rectangle(img, (x1,y1),(x2,y2), color=col, thickness=1)

        cv.imshow("img", img)
        cv.waitKey(0)