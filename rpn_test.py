import sys
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from torch.utils.data import DataLoader, Subset
import cv2 as cv
import torch

DEBUG = True
SHOW = False

name = 'jsons\\PARSEDOUTPUT-creeper1.json'

# Loading only one image
ds = torch.load("data\\datasets\\minedata_classifier_local.dtst")
ds = Subset(ds, [0])
dl = DataLoader(ds, batch_size=1, pin_memory=True)

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

if SHOW:
    img = img.permute(0, 2, 3, 1)[0, ...].numpy()
    cv.imshow("img", img)
    cv.waitKey()


