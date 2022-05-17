import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
from ClassLoader import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
import cv2
from backCNN import BackboneCNN

trans = transforms.Compose(
(
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.95, 1.05)))
)




ds = ClassData(["jsons\\OUTPUT-creeper1.json"], transform=trans)

split = int(ds.shape()[0] * 0.8)
train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, ds.shape()[0])))

train_load = DataLoader(train, batch_size=32, shuffle=True)
val_load =   DataLoader(val, batch_size=32, shuffle=True)

model = BackboneCNN()

optimizer = torch.optim.SGD(params=model.parameters() ,lr = 0.001, momentum=0.9, weight_decay=1e-2)
best_loss = torch.inf
best_state = model.state_dict()

loss_funct = torch.nn.CrossEntropyLoss()

load = False
store = True

if load:
    model.load_state_dict(torch.load("./BackCNN_best_weights.pth"))

i = 0
max_c = 10
while True:

    model.train()
    for img, label in train_load:
        
        preds = model.forward(img)
        loss = loss_funct.forward(preds, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Minibatch!")
    model.eval()
    with torch.no_grad():
        for img, label in val_load:
            preds = model.forward(img)
            loss = loss_funct.forward(preds, label)

            total_loss = loss.item()
            total_correct = preds.argmax(axis=1).eq(label).sum().item()
            total = preds.shape[0]

            risk = total_loss / total
            accuracy = total_correct / total
            print(f"Epoch {i}: accuracy={accuracy:.5f}, risk={risk:.5f}")
        
        if loss < best_loss:
            best_loss = loss
            best_state = model.state_dict()
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training, best loss: {best_loss}.")
            break

    i += 1


if store:
    torch.save(best_state, "./BackCNN_best_weights.pth")