import cv2
import os
import random
import numpy as np

random.seed(42)

files = os.listdir("./imgs/frames_new")


indexes = random.choices(list(range(0, len(files))), k=144)

img = np.zeros((720,720,3), dtype=np.uint8)

t = 0
for y in range(16):
    for x in range(9):
        curr_img = cv2.imread("./imgs/frames_new/" + files[indexes[t]])
        frame = cv2.resize(curr_img,(80,45), interpolation=cv2.INTER_AREA)
        img[y*45:(y+1)*45, x*80:(x+1)*80] = frame
        t += 1

cv2.imshow("Final img", img)
cv2.waitKey(0)

cv2.imwrite("./docs/presentation/dtset_repr.png", img)
