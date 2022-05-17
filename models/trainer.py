import torch
from backCNN import BackboneCNN
import sys
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
from MineDataset import MineDatasetMultiTensor


def l2_norm_loss(anch, p, n, margin):
    dp = torch.sqrt(torch.pow(anch - p, 2).sum(axis = 1))
    dn = torch.sqrt(torch.pow(anch - n, 2).sum(axis = 1))
    return torch.clamp(dp - dn + margin, min=0.0)

# def cos_dist_loss(anch, p, n, margin):
#     dp = 1.0 - torch.diag(torch.tensordot(anch,p, dims=([1,2,3],[1,2,3]))) / torch.sqrt(torch.pow(p,2).sum(axis = (1,2,3)) * torch.pow(anch,2).sum(axis = (1,2,3)))
#     dn = 1.0 - torch.diag(torch.tensordot(anch,n, dims=([1,2,3],[1,2,3]))) / torch.sqrt(torch.pow(n,2).sum(axis = (1,2,3)) * torch.pow(anch,2).sum(axis = (1,2,3)))
#     return torch.clamp(dp - dn + margin, min=0.0)

def cos_dist_loss(anch, p, n, margin):
    dp = 1.0 - torch.diag(anch @ p.T) / torch.sqrt(torch.pow(p,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
    dn = 1.0 - torch.diag(anch @ n.T) / torch.sqrt(torch.pow(n,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
    return torch.clamp(dp - dn + margin, min=0.0)

class TripletLoss(torch.nn.Module):
    # Shamelessly stolen from https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, p, n):
        return cos_dist_loss(anchor, p, n, self.margin).sum()


load = False
store = True

# define the device for the computation
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

BATCH_SIZE = 128
    
dataset = MineDatasetMultiTensor(os.path.join("data", "datasets"), "mine-classes")

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

network = BackboneCNN().to(device)
    
if load:
    network.load_state_dict(torch.load("./BackCNN_best_weights.pth"))

optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

best_loss = torch.inf
best_state = network.state_dict()

loss_funct = torch.nn.CrossEntropyLoss()

i = 0
counter = 0
max_c = 10
MAX_ITERS = 5

cv2.namedWindow("Feature Map 1", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Feature Map 2", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Feature Map 3", cv2.WINDOW_NORMAL)

while True:
    i += 1
    j = 0

    network.train()

    for imgs, labels in iter(loader):
        # Train

        imgs = imgs.to(device=device)
        preds = network.forward(imgs)

        loss = loss_funct.forward(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        j += 1
        print(j)


    # Valid
    network.eval()
    with torch.no_grad():

        imgs, labels = next(iter(loader))

        imgs = imgs.to(device=device)
        
        preds, feature_maps = network.forward(imgs, True)

        loss = loss_funct.forward(preds, labels)
    
        total_loss = loss.item()
        total_correct = preds.argmax(axis=1).eq(labels).sum().item()
        total = preds.shape[0]
        
        risk = total_loss / total
        accuracy = total_correct / total
        print(f"Epoch {i}: accuracy={accuracy:.5f}, risk={risk:.5f}")

        print(preds.argmax(axis=1))
        print(labels)

        cv2.imshow("Feature Map 1", torch.clamp(feature_maps[0][:3] * 255, min=0, max=255).type(torch.uint8).permute(1,2,0).numpy())
        # cv2.imshow("Feature Map 2", torch.clamp(feature_maps[0][3:6] * 255, min=0, max=255).type(torch.uint8).permute(1,2,0).numpy())
        # cv2.imshow("Feature Map 3", torch.clamp(feature_maps[0][6:9] * 255, min=0, max=255).type(torch.uint8).permute(1,2,0).numpy())

        cv2.waitKey()

        if loss < best_loss:
            best_loss = loss
            best_state = network.state_dict()
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training...")
            break


if store:
    torch.save(best_state, "./BackCNN_best_weights.pth")

cv2.destroyAllWindows()
    