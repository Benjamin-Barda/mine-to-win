from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C


"""
Anchors
"""
# Scale list to generate anchors
__C.ANCHOR_SCALES = [.25, .5, 1, 2]

# Ratio list to generate anchors
__C.ANCHOR_RATIOS = [.5, 1, 2]

# # Scale list to generate anchors
# __C.ANCHOR_SCALES = [.7, 1, 1.43]

# # Ratio list to generate anchors
# __C.ANCHOR_RATIOS = [.5, .7, 1, 1.43, 2]

__C.A = len(__C.ANCHOR_SCALES) * len(__C.ANCHOR_RATIOS)

# TODO: Remove after tesing
__C.COMPUTED_ANCHORS = np.array([[ -83.,  -39.,  100.,   56.],
                                [-175.,  -87.,  192.,  104.],
                                [-359., -183.,  376.,  200.],
                                [ -55.,  -55.,   72.,   72.],
                                [-119., -119.,  136.,  136.],
                                [-247., -247.,  264.,  264.],
                                [ -35.,  -79.,   52.,   96.],
                                [ -79., -167.,   96.,  184.],
                                [-167., -343.,  184.,  360.]])

"""
RegionProposalNetwork
"""

# Number of dimension in out from the first Conv+ReLU layer in the region proposal
__C.BASE_CONV_OUT_SIZE = 124

# Total stride of the backbone network
__C.FEATURE_STRIDE = 18

"""
RegionProposalNetwork.RegionProposalLayer
"""
# in train how many ROI to retain before NMS
__C.PRE_NMS_TRAIN  = 1000
# in train how many ROI to retain after NMS
__C.POST_NMS_TRAIN = 1000
# in testing how many ROI to retain before NMS
__C.PRE_NMS_TEST   = 2000
# in testing how many ROI to retain after NMS
__C.POST_NMS_TEST  = 1000
# Threshhold for nms
__C.NMS_THRESHOLD = .1

"""
CUDA Device
"""
import torch
__C.DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"
