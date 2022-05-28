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
ds = torch.load("data\\datasets\\minedata_actual_local.dtst")
dl = DataLoader(ds, batch_size=bs, pin_memory=True, shuffle=True)

state_extractor, state_rpn = torch.load("./MineRPN_best_weights.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(state_extractor)
lossfn = RPNLoss()

extractor.eval()

with torch.no_grad():
    img, lbl, bounds = ds[100]
    img = img[None, ...]
    base_feat_map = extractor.forward(img)
_, inDim, hh, ww, = base_feat_map.size()

rpn = _rpn(inDim).to(("cpu"))
rpn.load_state_dict(state_rpn)

score, reg, rois_index = rpn(base_feat_map, img.shape[-2:])

img = img.permute(0, 2, 3, 1)[0, ...].numpy()
img = np.ascontiguousarray(img)
cv.imshow("orig img", img)

print(score.shape)
for indx, an in enumerate(rois_index[0][0]):
    col = (0,255,0)
    if score[0][indx] < .5:
        col = (0,0,255)
        break
    x, y, w, h = an

    x1,y1,x2,y2 = int(x.item() - w.item()//2), int(y.item() - h.item()//2), int(x.item() + w.item()//2), int(y.item() + h.item()//2)

    cv.rectangle(img, (x1,y1),(x2,y2), color=col, thickness=1)

cv.namedWindow("img", cv.WINDOW_NORMAL)
cv.imshow("img", img)
cv.waitKey(0)