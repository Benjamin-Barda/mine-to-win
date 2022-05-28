import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.loss_func import RPNLoss
from models.regionProposal.utils.anchorUtils import *
from torch.utils.data import DataLoader, Subset
import cv2 as cv
import torch
from data.ClassData import ClassData
from torch.backends import cuda
import random


cuda.benchmark = True
SHOW = True

ds = torch.load("data\\datasets\minedata_actual_local.dtst")

device = "cuda" if torch.cuda.is_available() else "cpu"

split = int(ds.shape()[0] * 0.75)
train, val, test = Subset(ds, list(range(split))), Subset(ds, list(range(split, int(ds.shape()[0] * 0.9)))), Subset(ds, list(range(int(ds.shape()[0] * 0.9), int(ds.shape()[0]))))

BATCH_SIZE = 1
ANCHORS_HALF_BATCH_SIZE = 64

train_load = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
val_load =   DataLoader(val, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
test_load =   DataLoader(test, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(device)
extractor.load_state_dict(torch.load("./models/extractor/backbone_trained_weights.pth", map_location=device))
rpn = _rpn(240).to(device)

load = True
store = True

if load:
    state_extractor, state_rpn = torch.load("./MineRPN_best_weights.pth", map_location=device)
    extractor.load_state_dict(state_extractor)
    rpn.load_state_dict(state_rpn)

params = list(extractor.parameters()) + list(rpn.parameters())
optimizer = torch.optim.AdamW(params=params, lr = 0.0001, amsgrad=True)

best_risk = torch.inf
best_state = (extractor.state_dict(), rpn.state_dict())

loss_funct = RPNLoss()

i = 0
counter = 0
max_c = 15

tot_minibatch = np.ceil(split / BATCH_SIZE)
tot_minibatch_val = np.ceil((split * 0.2) / BATCH_SIZE)

while True:
    extractor.train()
    rpn.train()
    ds.set_train_mode(True)

    for indx, (img, label, bounds) in enumerate(train_load):
        
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        bounds = bounds[:, :, 1:]
        bounds = bounds.to(device, non_blocking=True)
        
        base_feat_map = extractor.forward(img)
        _, inDim, hh, ww, = base_feat_map.size()

        score, reg, rois_index = rpn(base_feat_map, img.shape[-2:])

        labels, values = label_anchors(bounds, hh, ww, rpn.anchors)

        values = values.permute(0,2,1)

        score = score[0][rois_index[0][1].tolist()]
        reg = reg[0][rois_index[0][1].tolist()]
        labels = labels[0][rois_index[0][1].tolist()].type(torch.int64)
        values = values[0][rois_index[0][1].tolist()]

        positives = list()
        negatives = list()
        for j,label in enumerate(labels):
            if label == 1:
                positives.append(j)
            elif label == -1:
                negatives.append(j)
                labels[j] = 0
        
        random.shuffle(positives)
        random.shuffle(negatives)

        to_use = []
        if len(positives) >= ANCHORS_HALF_BATCH_SIZE:
            to_use += positives[:ANCHORS_HALF_BATCH_SIZE]
            to_use += negatives[:ANCHORS_HALF_BATCH_SIZE]
        else:
            to_use = positives
            to_use += negatives[:ANCHORS_HALF_BATCH_SIZE * 2 - len(to_use)]

        loss = loss_funct.forward(score[to_use], reg[to_use], labels[to_use], values[to_use])
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {i}, {(100 * indx / tot_minibatch):.3f}%", end="\r")

    extractor.eval()
    rpn.eval()
    ds.set_train_mode(False)
    
    with torch.no_grad():
        total_loss, total_correct, total_sqrd_error, total = 0,0,0,0
        
        for img, label, bounds in val_load:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            bounds = bounds[:, :, 1:]
            bounds = bounds.to(device, non_blocking=True)
            
            base_feat_map = extractor.forward(img)
            _, inDim, hh, ww, = base_feat_map.size()

            score, reg, rois_index = rpn(base_feat_map, img.shape[-2:])

            labels, values = label_anchors(bounds, hh, ww, rpn.anchors)

            values = values.permute(0,2,1)

            score = score[0][rois_index[0][1].tolist()]
            reg = reg[0][rois_index[0][1].tolist()]
            labels = labels[0][rois_index[0][1].tolist()].type(torch.int64)
            values = values[0][rois_index[0][1].tolist()]

            positives = list()
            negatives = list()
            for j,label in enumerate(labels):
                if label == 1:
                    positives.append(j)
                elif label == -1:
                    negatives.append(j)
                    labels[j] = 0
            
            random.shuffle(positives)
            random.shuffle(negatives)

            to_use = []
            if len(positives) >= ANCHORS_HALF_BATCH_SIZE:
                to_use += positives[:ANCHORS_HALF_BATCH_SIZE]
                to_use += negatives[:ANCHORS_HALF_BATCH_SIZE]
            else:
                to_use = positives
                to_use += negatives[:ANCHORS_HALF_BATCH_SIZE * 2 - len(to_use)]

            loss = loss_funct.forward(score[to_use], reg[to_use], labels[to_use], values[to_use])

            total += len(to_use)
            total_loss += loss.item()
            total_correct += torch.where(score[to_use] < .5, 0, 1).eq(labels[to_use]).sum().item()
            total_sqrd_error += (torch.pow(reg[to_use] - values[to_use], 2)).sum().item()
            
            
        risk = total_loss / tot_minibatch_val
        accuracy = total_correct / total
        print(f"Epoch {i}: accuracy={accuracy:.5f}, sqrd_error={total_sqrd_error/(total*4):.5f}, risk={risk:.5f}")
        
        if risk < best_risk:
            best_risk = risk
            best_state = (extractor.state_dict(), rpn.state_dict())
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training, best risk: {best_risk}.")
            break
    break

if store:
    torch.save(best_state, "./MineRPN_best_weights.pth")