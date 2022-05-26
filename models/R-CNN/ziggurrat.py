import sys
import os
import torch
import torch.nn as nn
import numpy as np


class Monster(nn.Module):

    def __init__(self, extractor, rpn, extractor_weights=None, rpn_weights=None):
        super(Monster, self).__init__()

        self.extractor = extractor
        self.rpn = rpn

    def forward(self, x):
        base_feat_map = self.extractor(x)

        score, offsets, rois = self.rpn(base_feat_map)

        return rois
