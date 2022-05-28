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
        img_size = x.shape[-2:]

        base_feat_map = self.extractor(x)

        score, offsets, rois_x_index = self.rpn(base_feat_map, img_size)

        return rois_x_index[0]
