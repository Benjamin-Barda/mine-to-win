from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C 


# Ratios and scales for the anchors
__C.ANCHOR_SCALES = [8, 16, 32]
__C.ANCHOR_RATIOS = [.5, 1, 2]
__C.COMPUTED_ANCHORS = np.array([[ -83.,  -39.,  100.,   56.],
                                [-175.,  -87.,  192.,  104.],
                                [-359., -183.,  376.,  200.],
                                [ -55.,  -55.,   72.,   72.],
                                [-119., -119.,  136.,  136.],
                                [-247., -247.,  264.,  264.],
                                [ -35.,  -79.,   52.,   96.],
                                [ -79., -167.,   96.,  184.],
                                [-167., -343.,  184.,  360.]])

# RPN 
__C.BASE_CONV_OUT_SIZE = 124