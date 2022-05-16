import torch
import pandas as pd
from torch.utils import data

class ClassLoader(data.Dataset):


    def __init__(self, sources = [],  transform = None):

        df_iter = (pd.read_json(src) for src in sources)

        df = pd.concat(df_iter)
            