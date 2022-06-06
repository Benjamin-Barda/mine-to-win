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

state_extractor, state_rpn = torch.load("./weights_too_heavy/MineRPN_best_weights.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(state_extractor)
lossfn = RPNLoss()

_, inDim, hh, ww, = extractor(ds[0][0][None, ...]).size()

rpn = _rpn(inDim).to(("cpu"))
rpn.load_state_dict(state_rpn)

extractor.eval()
rpn.eval()

model_parameters = filter(lambda p: p.requires_grad, rpn.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(params)

cv.namedWindow("img", cv.WINDOW_NORMAL)

with torch.no_grad():
    
    for i in range(100, 115):

        img, lbl, elem = ds[i]
        img = img[None, ...]
        base_feat_map = extractor(img)
        bounds, b_label = ds.getvertex(elem)
        score, ts, rois, nms_indexes = rpn(base_feat_map, img.shape[-2:], False)

        rois = rois[0][nms_indexes]
        score = score[0][nms_indexes]

        img = img.permute(0, 2, 3, 1)[0, ...].numpy()
        img = np.ascontiguousarray(img)
        cv.imshow("orig img", img)
        for indx, an in enumerate(rois):
            
            col = (0,255,0)
            if score[indx] < .8:
                col = (0,0,255)
                continue

            x,y,w,h = an.int()

            x1,y1,x2,y2 = int(x.item() - w.item()//2), int(y.item() - h.item()//2), int(x.item() + w.item()//2), int(y.item() + h.item()//2)
            cv.rectangle(img, (x1,y1),(x2,y2), color=col, thickness=1)

        cv.imshow("img", img)
        cv.waitKey(0)