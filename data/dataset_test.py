from MineDataset import MineDatasetMulti
import os
import cv2

MOB_MAP  =  {
    ord('1') : ('Pig', (205, 170, 230)),
    ord('2') : ('Cow', (50, 70, 120)),
    ord('3') : ('Chicken', (5, 5, 200)),
    ord('4') : ('Sheep', (125, 125, 121)),
    ord('5') : ('Zombie', (50, 120, 20)),
    ord('6') : ('Skeleton', (225, 225, 225)),
    ord('7') : ('Creeper', (50, 168, 58)),
    ord('8') : ('Spider', (20, 16, 26)),
    ord('9') : ('Wolf', (255, 119, 82)),
    ord('0') : ('Slime', (85, 255, 82)),
}


d = MineDatasetMulti(os.path.join("data", "datasets"), "mine-classes")

index = [300, 1002]
choose = 1

imgs, labels = d[index]
print(labels.iloc[choose])

rectList = []

for key, value in MOB_MAP.items():
    for lst in labels.iloc[choose][value[0]]:
        rectList.append((lst[0], lst[1], lst[2], lst[3], value[1]))

img = imgs.permute(0,2,3,1).numpy()[choose]

for rect in rectList: 
    mx, my, w, h, cols = rect
    cols = (cols[0]/255, cols[1]/255, cols[2]/255)
    cv2.rectangle(img, pt1 = (mx - w // 2, my - h // 2), pt2 = (mx + w // 2, my + h // 2), color = cols, thickness=2)

cv2.imshow("Image", img)
cv2.waitKey(0)