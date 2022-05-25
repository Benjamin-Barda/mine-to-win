import torch
import numpy as np


def generate_anchors(base, ratios, scales):
    """
    args :
        base   : int : W and H of windows
        ratios : array : ratios to apply to base
        scales : array : scales to apply to base
    return :
        anchors Tensor : (A,4) {x_center, y_center, Width, height}
    """

    return torch.from_numpy(np.vstack(
        [[0, 0, base * scale * np.sqrt(ratio), base * scale * np.sqrt(1. / ratio)] for scale in scales for ratio in
         ratios]
    ))


def centr2corner(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    ul_x = x - (w / 2)
    ul_y = y - (h / 2)

    br_x = x + (w / 2)
    br_y = y + (h / 2)

    return ul_x, ul_y, br_x, br_y
