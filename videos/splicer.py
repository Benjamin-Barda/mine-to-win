import numpy as np
import cv2

name = "world2"
i = 0
k = 0
cap = cv2.VideoCapture(name + ".mp4")

ret, frame = cap.read()

while ret:
    if i % 30 == 0:
        frame = cv2.resize(frame,(640,360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(name +"/" + name + f"-{k:05d}.png", frame)
        k += 1
    i += 1
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()