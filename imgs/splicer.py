import numpy as np
import cv2

names = ["test"]

for j in range(2, 3):
    print(j)
    for name in names:
        i = 0
        k = 0
        nname = name + f"{j}"
        print(nname)
        cap = cv2.VideoCapture("imgs/" + nname + ".mp4")

        ret, frame = cap.read()

        while ret:
            if i % 30 == 0:
                frame = cv2.resize(frame,(318,318), interpolation=cv2.INTER_AREA)
                cv2.imwrite("imgs/frames_new/" + nname + f"-{k:05d}.png", frame)
                k += 1
            i += 1
            ret, frame = cap.read()

        cap.release()