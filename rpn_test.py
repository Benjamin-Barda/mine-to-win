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

name = 'jsons\\PARSEDOUTPUT-creeper1.json'

# Loading only one image
ds = torch.load("data\\datasets\\minedata_classifier_local.dtst")
ds = Subset(ds, [x for x in range(bs)])
dl = DataLoader(ds, batch_size=bs, pin_memory=True)

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(torch.load("./BackCNN_deep3_best_weights3.pth", map_location=torch.device('cpu')))
lossfn = torch.nn.CrossEntropyLoss()

extractor.eval()

with torch.no_grad():
    for img, lbl in dl:
        img = img
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
    img = img.permute(0, 2, 3, 1)[0, ...].numpy()
    img = np.ascontiguousarray(img)
    for an in anchors:
        x, y, h, w = an

        x1,y1,x2,y2 = x.item() - w.item()//2, y.item() - h.item()//2, x.item() + w.item()//2, y.item() + h.item()//2

        #print(x1,y1,x2,y2)

        cv.rectangle(img, (x1,y1),(x2,y2), color=(0,0,255), thickness=1)

    cv.imshow("img", img)
    cv.waitKey()
