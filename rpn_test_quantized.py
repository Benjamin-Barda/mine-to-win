
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn
from models.regionProposal.loss_func import RPNLoss
from models.regionProposal.utils.anchorUtils import *
from torch.utils.data import DataLoader
from data.ClassData import ClassData

import torch
import torch.quantization

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

DEBUG = False
SHOW = True

bs = 1

# Loading only one image
ds = torch.load("data\\datasets\\minedata_compressed_local.dtst")
dl = DataLoader(ds, batch_size=bs, pin_memory=False, shuffle=True)

state_extractor, state_rpn = torch.load("weights/MineRPN_best_weights.pth", map_location=torch.device('cpu'))

# Model initialized with flag so after the last conv layer return the featmap
extractor = BackboneCNN(is_in_rpn=True).to(("cpu"))
extractor.load_state_dict(state_extractor)
lossfn = RPNLoss()

_, inDim, hh, ww, = extractor(ds[0][0][None, ...]).size()

rpn = _rpn(inDim).to(("cpu"))
rpn.load_state_dict(state_rpn)

extractor.eval()
rpn.eval()

print_size_of_model(rpn, label="rpn before")

rpn_q = torch.quantization.quantize_dynamic(
    rpn,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

print_size_of_model(rpn_q, label="rpn after")
print_size_of_model(extractor, label="extractor after")

best_state = (extractor.state_dict(), rpn_q.state_dict())
torch.save(best_state, "weights/MineRPN_quantized_weights.pth")