import pickle
import numpy as np

pairs = None

with open("score_label_pairs_roc_conv.pickle", "rb") as handle:
    pairs = pickle.load(handle)

pairs.sort()

pos = 0
neg = 0

for score, label in pairs:
    if label == 0:
        neg += 1
    else:
        pos += 1

neg_left = neg
pos_left = pos

values_pairs = list()

for score, label in pairs:
    values_pairs.append((score, neg_left / neg, pos_left / pos))
    if label == 0:
        neg_left -= 1
    else:
        pos_left -= 1

values_pairs = np.array(values_pairs).T

from matplotlib import pyplot as plt

lspace = np.linspace(0, 1.0, 11)

plt.plot(values_pairs[0], values_pairs[1], color="blue", label="FPR")
plt.plot(values_pairs[0], values_pairs[2], color="red", label="TPR")
plt.xlabel("Threshold")
plt.ylabel("TPR & FPR")
plt.xticks(np.linspace(0, 1.0, 11))
plt.yticks(np.linspace(0, 1.0, 11))
plt.grid()
plt.legend()
plt.show()