from msilib.schema import Class
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
from ClassData import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
import cv2
from backCNN import BackboneCNN
from torch.backends import cuda

import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

trans = transforms.ToTensor()

def plot_confusion(lab, pred):
    cmat = confusion_matrix(lab, pred)
    df_cm = pd.DataFrame(cmat, index = [i for i in ["Nothing", "Creeper", "Pig"]],
                    columns = [i for i in ["Nothing", "Creeper", "Pig"]])

    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True, linewidths=0.5)
    ax.set_xlabel('Targets', fontsize=14)
    ax.set_ylabel('Predictions', fontsize=14)
    plt.show()

data = ClassData(["jsons\\PARSEDOUTPUT-2022-05.json"], transform = trans)
dl = DataLoader(data, pin_memory=True, batch_size=2)

torch.save(data, "data.dtset")

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model = BackboneCNN().to("cuda")
model.load_state_dict(torch.load("./BackCNN_deep_best_weights.pth"))
lossfn = torch.nn.CrossEntropyLoss()

print(model)

model.conv6.register_forward_hook(get_features('feats'))

cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)
cv2.namedWindow("Map", cv2.WINDOW_NORMAL)

features = {}
total_l = []
total_p = []

model.eval()
print(model)

with torch.no_grad():
    for img, label in dl:
        
        img, label = img.to("cuda"), label.to("cuda")
        preds = model.forward(img)
        
        label = label.to("cpu")
        preds = preds.to("cpu")
        
        total_l.append(label.numpy())
        total_p.append(preds.numpy())
        

        for indx, im in enumerate(img):
            cv2.imshow("Sample", im.to("cpu").permute(1,2,0).numpy())
            x = features['feats']
            cv2.imshow("Map", features['feats'][indx].permute(1,2,0).cpu().numpy())
            cv2.waitKey(-1)

        
        label = label.to("cpu")
        print(preds.argmax(axis=1), label)
        
pred, lab = np.concatenate(total_p).argmax(axis = 1), np.concatenate(total_l)  

plot_confusion(lab, pred)

guesses = np.equal(pred, (lab))
acc = guesses.sum() / len(lab)

print(f"Guesses: {guesses}\n Accuracy: {100*acc:.3f}%")
        
cv2.destroyAllWindows()