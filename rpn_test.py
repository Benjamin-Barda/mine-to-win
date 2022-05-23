import sys, os
from models.extractor.backCNN import BackboneCNN
from models.regionProposal.rpn import _rpn 
import torch



# 
name = 'jsons\\PARSEDOUTPUT-creeper1.json'

ds = torch.load("data\\datasets\\minedata_classifier_local.dtst")