import sys
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.utils.anchorUtils import  *
from torch.utils.data import DataLoader, Subset
import cv2 as cv
import torch

DEBUG = False
SHOW = False

bs = 1

name = 'jsons\\PARSEDOUTPUT-creeper1.json'

# Loading only one image
ds = torch.load("data\\datasets\\minedata_classifier_local.dtst")
ds = Subset(ds, [x for x in range(bs)])
dl = DataLoader(ds, batch_size=bs, pin_memory=True)

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(torch.load("./bestPTH/BackCNN_best_weights.pth", map_location=torch.device('cpu')))
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
anchors = rpn_conv_out[-1].reshape(2808,4)
print(anchors.shape)

if SHOW:
    img = img.permute(0, 2, 3, 1)[0, ...].numpy()
    for an in anchors:
        x,y,w,z =  centr2corner(an)
        x = int(x)
        y = int(y)
        w = int(w)
        z = int(z)


    cv.imshow("img", img)
    cv.waitKey()


