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
        [0, 0, base * scale * np.sqrt(ratio), base * scale * np.sqrt(1. / ratio)] for scale in scales for ratio in
        ratios
    ))