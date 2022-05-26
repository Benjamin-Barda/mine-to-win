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


def center2corner(tensor_batch):
    n_batch, n_box, _ = tensor_batch.shape

    x = tensor_batch[:, :, 0::4]
    y = tensor_batch[:, :, 1::4]
    w = tensor_batch[:, :, 2::4]
    h = tensor_batch[:, :, 3::4]

    trans = tensor_batch.clone()

    trans[:, :, 0::4] = x - (w / 2)
    trans[:, :, 1::4] = y - (h / 2)
    trans[:, :, 2::4] = x + (w / 2)
    trans[:, :, 3::4] = y + (h / 2)

    return trans


def corner2center(tensor_batch):
    n_batch, n_box, _ = tensor_batch.shape

    x0 = tensor_batch[:, :, 0::4]
    y0 = tensor_batch[:, :, 1::4]
    x1 = tensor_batch[:, :, 2::4]
    y1 = tensor_batch[:, :, 3::4]

    trans = tensor_batch.clone()

    # x_ctr
    trans[:, :, 0::4] = (x1 - x0) / 2
    # y_ctr
    trans[:, :, 1::4] = (y1 - y0) / 2
    # Width
    trans[:, :, 2::4] = x1 - x0
    # Height
    trans[:, :, 3::4] = y1 - y0
