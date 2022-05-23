import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from data.ClassData import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
import cv2
from models.extractor import BackboneCNN
from torch.backends import cuda

import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion(lab, pred):
    cmat = confusion_matrix(lab, pred)
    df_cm = pd.DataFrame(cmat, index = [i for i in ["Nothing", "Creeper", "Pig"]],
                    columns = [i for i in ["Nothing", "Creeper", "Pig"]])

    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True, linewidths=0.5)
    ax.set_xlabel('Targets', fontsize=14)
    ax.set_ylabel('Predictions', fontsize=14)
    plt.show()

cuda.benchmark = True

trans = transforms.Compose((
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.55, 1.45), contrast=(0.95, 1.05), saturation=(0.99, 1.01))
))

names = [f"jsons\\PARSEDOUTPUT-creeper{i}.json" for i in range(1, 9)]
names += [f"jsons\\PARSEDOUTPUT-pig{i}.json" for i in range(1, 9)]
names += [f"jsons\\PARSEDOUTPUT-null{i}.json" for i in range(1, 9)]
names.append("jsons\\PARSEDOUTPUT-seanull1.json")

ds = ClassData(names, transform=trans)
print(ds.shape())

split = int(ds.shape()[0] * 0.8)
train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, ds.shape()[0])))

BATCH_SIZE = 32

train_load = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
val_load =   DataLoader(val, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

model = BackboneCNN().to("cuda", non_blocking=True)

optimizer = torch.optim.AdamW(params=model.parameters() ,lr = 0.0002, amsgrad=True)
best_risk = torch.inf
best_state = model.state_dict()

loss_funct = torch.nn.CrossEntropyLoss()

load = True
store = True

if load:
    model.load_state_dict(torch.load("./BackCNN_best_weights.pth"))

i = 0
max_c = 5

tot_minibatch = np.ceil(split / BATCH_SIZE)
tot_minibatch_val = np.ceil((split * 0.2) / BATCH_SIZE)

while True:

    model.train()
    for indx, (img, label) in enumerate(train_load):
        
        img = img.to("cuda", non_blocking=True)
        label = label.to("cuda", non_blocking=True)
         
        preds = model.forward(img)
        loss = loss_funct.forward(preds, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {i}, {(100 * indx / tot_minibatch):.3f}%", end = "\r")
        
    model.eval()
    with torch.no_grad():
        total_loss, total_correct, total = 0,0,0
        for img, label in val_load:
            img = img.to("cuda", non_blocking=True)
            label = label.to("cuda", non_blocking=True)
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
    torch.save(best_state, "./BackCNN_finetuned_best_weights.pth")