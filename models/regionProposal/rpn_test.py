import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.extractor import BackboneCNN
from models.regionProposal.rpn import _rpn 
from data.ClassData import ClassData

name = 'jsons\\PARSEDOUTPUT-creeper1.json'

ds = ClassData(name)