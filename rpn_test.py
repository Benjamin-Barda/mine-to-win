import sys
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
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

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(torch.load("./BackCNN_deep3_best_weights3.pth", map_location=torch.device('cpu')))
lossfn = torch.nn.CrossEntropyLoss()

extractor.eval()

with torch.no_grad():
    img, lbl, bounds = ds[400]
    print(bounds)
    img = img[None, ...]
    img_size = img.shape[-2:]
    base_feat_map = extractor.forward(img)
_, inDim, hh, ww, = base_feat_map.size()

if DEBUG:
    print(f"shape of feature map before rpn= {base_feat_map.shape}")
    print(f"shape of the image = {img_size}")

rpn = _rpn(inDim)
rpn_conv_out = rpn(base_feat_map, img_size)

if SHOW:
    anchors = rpn_conv_out[-1].reshape(-1, 4).type(torch.int32)
    labels, values = label_anchors([bounds], hh, ww, rpn.anchors)
    labels = labels[0]
    values = values[0]
    img = img.permute(0, 2, 3, 1)[0, ...].numpy()
    img = np.ascontiguousarray(img)
    cv.imshow("orig img", img)
    for indx, an in enumerate(anchors):
        col = (0,255,0)
        if labels[indx] == -1:
            col = (0,0,255)
        elif labels[indx] == 0:
            col = (255,0,0)
            continue
        else:
            print(values.T[indx])
        x, y, h, w = an

        x1,y1,x2,y2 = x.item() - w.item()//2, y.item() - h.item()//2, x.item() + w.item()//2, y.item() + h.item()//2

        #print(x1,y1,x2,y2)

        cv.rectangle(img, (x1,y1),(x2,y2), color=col, thickness=1)

    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.imshow("img", img)
    cv.waitKey(0)
