import cv2 as cv
from cv2 import threshold
import torch
from torch.utils.data import DataLoader
import numpy as np
from data.ClassData import ClassData
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.utils.anchorUtils import *

threshold = 0.81 # Change this to modify TPR and FPR!


# Loading only one image
ds = torch.load("data/datasets/minedata_compressed_test.dtst")
dl = DataLoader(ds, batch_size=1, pin_memory=True, shuffle=True)

state_extractor, state_rpn = torch.load("./models/weights/MineRPN_weights_quantized.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(state_extractor)
extractor.eval()
extractor = torch.jit.freeze(torch.jit.script(extractor))

_, inDim, hh, ww, = extractor(ds[0][0][None, ...]).size()

rpn = _rpn(inDim).to(("cpu"))
rpn = torch.quantization.quantize_dynamic(
    rpn,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

rpn.load_state_dict(state_rpn)

rpn.eval()


cv.namedWindow("img", cv.WINDOW_NORMAL)

# score_label_pairs = []

with torch.no_grad():
    
    j = 0

    for img, lbl, elem in ds:

        j += 1

        img = img[None, ...]
        base_feat_map = extractor(img)
        bounds, b_label = ds.getvertex(elem)
        score, ts, rois, nms_indexes = rpn(base_feat_map, img.shape[-2:], False)

        rois = rois[0][nms_indexes]
        score = score[0][nms_indexes]

        # bounds, b_label = ds.getvertex(elem)

        # labels, _ = label_anchors(bounds, hh, ww, rpn.anchors, img.shape[-2:])
        # labels = labels[nms_indexes]

        # labels = torch.clip(labels, min = 0)

        # score = score.tolist()
        # labels = labels.tolist()

        # for i, val in enumerate(labels):
        #     if score[i] < .81:
        #         score_label_pairs.append((0,val))
        #     else:
        #         score_label_pairs.append((1,val))

        img = img.permute(0, 2, 3, 1)[0, ...].numpy()
        img = np.ascontiguousarray(img)
        
        cv.imshow("orig img", img)
        for indx, an in enumerate(rois):
            
            col = (0,255,0)
            if score[indx] < threshold:
                col = (0,0,255)
                continue

            x,y,w,h = an.int()

            x1,y1,x2,y2 = int(x.item() - w.item()//2), int(y.item() - h.item()//2), int(x.item() + w.item()//2), int(y.item() + h.item()//2)
            cv.rectangle(img, (x1,y1),(x2,y2), color=col, thickness=1)

        cv.imshow("img", img)
        cv.waitKey(0)

# import pickle


# with open("pred_label_pairs_misc.pickle", "wb") as handle:
#     handle.write(pickle.dumps(score_label_pairs))
#     handle.close()