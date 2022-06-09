import sys
import os
import torch
from torch.backends import cuda
from torch.utils.data import DataLoader, Subset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.extractor.backCNN import BackboneCNN
from data.ClassData import ClassData
from models.regionProposal.rpn import _rpn # EITHER rpn OR rpn_conv
from models.regionProposal.loss_func import RPNLoss
from models.regionProposal.utils.anchorUtils import *

def main():

    torch.set_default_dtype(torch.float32)
    cuda.benchmark = True

    ds = torch.load("data/datasets/minedata_compressed.dtst")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = int(ds.shape()[0] * 0.8)
    train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, int(ds.shape()[0]))))

    BATCH_SIZE = 32
    ANCHORS_HALF_BATCH_SIZE = 1
    THREADS = 4
    
    train_load = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True, num_workers = THREADS)
    val_load =   DataLoader(val, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True, num_workers = THREADS)

    # Model initialized with flag so after the last conv layer return the featmap
    extractor = BackboneCNN(is_in_rpn=True).to(device)
    extractor.load_state_dict(torch.load("./models/weights/BackCNN_weights.pth", map_location=device))
    extractor = torch.jit.script(extractor)
    rpn = _rpn(240, device=device).to(device)


    load = False # We didn't upload 3 gbs of weights
    store = True

    if load:
        state_extractor, state_rpn = torch.load("./models/weights/MineRPN_weights.pth", map_location=device)
        extractor.load_state_dict(state_extractor)
        rpn.load_state_dict(state_rpn)

    params = list(extractor.parameters()) + list(rpn.parameters())
    optimizer = torch.optim.SGD(params=params, lr = 1e-5, momentum=.9)

    best_risk = torch.inf
    best_state = (extractor.state_dict(), rpn.state_dict())

    loss_funct = torch.jit.script(RPNLoss())

    i = 0
    counter = 0
    max_c = 2

    tot_minibatch = np.ceil(split / BATCH_SIZE)
    tot_minibatch_val = np.ceil((ds.shape()[0] * 0.2) / BATCH_SIZE)

    print(len(val))
    
    if device == "cuda":
        ds.tocuda()

    while True:
        extractor.train()
        rpn.train()
        ds.set_train_mode(True)

        for indx, (img, label, elements) in enumerate(train_load):
            
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            #elements = elements.to(device, non_blocking=True)
            loss = 0
            
            base_feat_map = extractor.forward(img)
            _, _, hh, ww, = base_feat_map.size()

            score, ts, _, _ = rpn(base_feat_map, img.shape[-2:])
                        
            for elem_indx, elem in enumerate(elements):
                
                bounds, b_label = ds.getvertex(elem)

                bounds.requires_grad = False
                b_label.requires_grad = False

                bounds = bounds.to(device, non_blocking=True)
                labels, values = label_anchors(bounds, hh, ww, rpn.anchors, img.shape[-2:])
                labels.requires_grad = False
                values.requires_grad = False
                labels = labels.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                values = values.permute(1,0)
                    

                positives = (labels > .999).nonzero().T.squeeze()
                positives.requires_grad = False
                negatives = (labels < -.999).nonzero().T.squeeze()
                negatives.requires_grad = False
                
                labels = torch.clip(labels, min = 0)
                
                if not len(positives.shape):
                    positives = torch.zeros((0))
                    
                if not len(negatives.shape):
                    negatives = torch.zeros((0))
                    
                positives = positives[torch.randperm(positives.shape[0])]
                negatives = negatives[torch.randperm(negatives.shape[0])]
                

                to_use = torch.empty((ANCHORS_HALF_BATCH_SIZE*2), dtype=torch.int64, device=device)
                to_use.requires_grad = False
                if positives.shape[0] >= ANCHORS_HALF_BATCH_SIZE:
                    to_use[:ANCHORS_HALF_BATCH_SIZE] = positives[:ANCHORS_HALF_BATCH_SIZE]
                    to_use[ANCHORS_HALF_BATCH_SIZE:] = negatives[:ANCHORS_HALF_BATCH_SIZE]
                else:
                    to_use[:positives.shape[0]] = positives
                    to_use[positives.shape[0]:] = negatives[:ANCHORS_HALF_BATCH_SIZE * 2 - positives.shape[0]]

                pred_score = score[elem_indx, to_use]
                pred_offset = ts[elem_indx, to_use]
                used_labels = labels[to_use]
                used_offset = values[to_use]
                used_labels.requires_grad = False
                used_offset.requires_grad = False

                loss += loss_funct.forward(pred_score, pred_offset, used_labels, used_offset)
                #print(loss_funct.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {i}, {(100 * indx / tot_minibatch):.3f}%", end="\r")

        extractor.eval()
        rpn.eval()
        ds.set_train_mode(False)
        
        with torch.no_grad():
            total_loss, total_correct, total_sqrd_error, total = 0,0,0,0
            total_pos, total_neg = 0, 0
            
            for indx, (img, label, elements) in enumerate(val_load):
            
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                elements = elements.to(device, non_blocking=True)
                loss = 0
                
                base_feat_map = extractor.forward(img)
                _, inDim, hh, ww, = base_feat_map.size()

                score, ts, _, _  = rpn(base_feat_map, img.shape[-2:])
                
                for elem_indx,elem in enumerate(elements):
                    
                    bounds, b_label = ds.getvertex(elem)
                    
                    bounds = bounds.to(device, non_blocking=True)
                    labels, values = label_anchors(bounds, hh, ww, rpn.anchors, img.shape[-2:])
                    labels = labels.to(device, non_blocking=True)
                    values = values.to(device, non_blocking=True)
                    values = values.permute(1,0)
                        

                    positives = (labels > .999).nonzero().T.squeeze()
                    negatives = (labels < -.999).nonzero().T.squeeze()
                    
                    labels = torch.clip(labels, min = 0)
                    
                    if not len(positives.shape):
                        positives = torch.zeros((0))
                        
                    if not len(negatives.shape):
                        negatives = torch.zeros((0))
                        
                    positives = positives[torch.randperm(positives.shape[0])]
                    negatives = negatives[torch.randperm(negatives.shape[0])]
                    
                    total_neg += negatives.shape[0]
                    total_pos += positives.shape[0]

                    to_use = torch.empty((ANCHORS_HALF_BATCH_SIZE*2), dtype=torch.int64, device=device)
                    if positives.shape[0] >= ANCHORS_HALF_BATCH_SIZE:
                        to_use[:ANCHORS_HALF_BATCH_SIZE] = positives[:ANCHORS_HALF_BATCH_SIZE]
                        to_use[ANCHORS_HALF_BATCH_SIZE:] = negatives[:ANCHORS_HALF_BATCH_SIZE]
                    else:
                        to_use[:positives.shape[0]] = positives
                        to_use[positives.shape[0]:] = negatives[:ANCHORS_HALF_BATCH_SIZE * 2 - positives.shape[0]]

                    pred_score = score[elem_indx, to_use]
                    pred_offset = ts[elem_indx, to_use]
                    used_labels = labels[to_use]
                    used_offset = values[to_use]

                    loss += loss_funct.forward(pred_score, pred_offset, used_labels, used_offset)

                    total += to_use.shape[0]
                    
                    total_correct += torch.where(pred_score < .5, 0, 1).eq(used_labels).sum().item()
                    total_sqrd_error += (torch.pow(pred_offset * used_labels[:, None].expand(-1, 4) - used_offset, 2)).sum().item()
                    
                    print(f"Epoch {i}, Validation, {(100 * indx / tot_minibatch_val):.3f}%", end="\r")
                    
                total_loss += loss.item()

            risk = total_loss / tot_minibatch_val
            accuracy = total_correct / total
            print(f"Epoch {i}: accuracy={accuracy:.5f}, sqrd_error={total_sqrd_error/(total*4):.5f}, risk={risk:.5f}")
            print(f"Total positive anchors: {total_pos}, Total negative anchors: {total_neg}, Ratio: {total_pos/total_neg:.5f}")
            if risk < best_risk:
                best_risk = risk
                best_state = (extractor.state_dict(), rpn.state_dict())
                counter = 0
            elif counter < max_c:
                counter += 1
            else:
                print(f"Worse loss reached, stopping training, best risk: {best_risk}.")
                break
            
            i+=1
    if store:
        torch.save(best_state, "./models/weights/MineRPN_weights_2.pth")

if __name__ == "__main__":
    main()