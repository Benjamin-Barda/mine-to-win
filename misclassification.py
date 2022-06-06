import pickle
import numpy as np

pairs = None

with open("score_label_pairs_test.pickle", "rb") as handle:
    pairs = pickle.load(handle)

pairs.sort()

tp = 0
fp = 0
tn = 0
fn = 0

for score, label in pairs:
    if score == 0:
        if label == -1:
            tn += 1
        else:
            fn += 1
    else:
        if label == 1:
            tp += 1
        else:
            fp += 1

print(tp, fp, tn, fn)