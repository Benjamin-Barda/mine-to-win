import sys, os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.backends import cuda
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.extractor.backCNN import BackboneCNN
from data.ClassData import ClassData

cuda.benchmark = True

ds = torch.load("data/datasets/minedata_compressed.dtst") # Works only with single objects in image, don't use the test!

device = "cuda" if torch.cuda.is_available() else "cpu"

split = int(ds.shape()[0] * 0.6)
train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, int(ds.shape()[0] * 0.8))))

BATCH_SIZE = 256

train_load = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
val_load =   DataLoader(val, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

model = BackboneCNN().to(device, non_blocking=True)

load = True
store = True

if load:
    model.load_state_dict(torch.load("models/weights/BackCNN_weights.pth", map_location=device))


optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.0002, amsgrad=True)

best_risk = torch.inf
best_state = model.state_dict()

loss_funct = torch.nn.CrossEntropyLoss()

i = 0
counter = 0
max_c = 15

tot_minibatch = np.ceil(split / BATCH_SIZE)
tot_minibatch_val = np.ceil((split * 0.2) / BATCH_SIZE)

while True:

    model.train()
    ds.set_train_mode(True)
    
    for indx, (img, label, _) in enumerate(train_load):
        
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        preds = model.forward(img)
        loss = loss_funct.forward(preds, label)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {i}, {(100 * indx / tot_minibatch):.3f}%", end="\r")
        
    model.eval()
    ds.set_train_mode(False)
    
    with torch.no_grad():
        total_loss, total_correct, total = 0,0,0
        
        for img, label, _ in val_load:
            
            #cv2.imshow("HI", img[0].permute(1,2,0).numpy())
            #cv2.waitKey(-1)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            total += img.shape[0]
            
            preds = model.forward(img)
            loss = loss_funct.forward(preds, label)

            total_loss += loss.item()
            total_correct += preds.argmax(axis=1).eq(label).sum().item()
            
            
        risk = total_loss / tot_minibatch_val
        accuracy = total_correct / total
        print(f"Epoch {i}: accuracy={accuracy:.5f}, risk={risk:.5f}")
        
        if risk < best_risk:
            best_risk = risk
            best_state = model.state_dict()
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training, best risk: {best_risk}.")
            break
    
        i += 1

if store:
    torch.save(best_state, "models/weights/BackCNN_weights_2.pth")