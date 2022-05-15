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


d = MineDatasetMulti(None, None, os.path.join("data", "datasets", "mine-classes.dtset"), True)

index = 300

v = d.data.iloc[index]
print(v)

rectList = []

for key, value in MOB_MAP.items():
    for lst in v[value[0]]:
        rectList.append((lst[0], lst[1], lst[2], lst[3], value[1]))
print(d.images.shape)
img = d.images[v["Image"]].numpy()

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for rect in rectList: 
    x0, y0, w, h, cols = rect
    cols = (cols[0]/255, cols[1]/255, cols[2]/255)
    cv2.rectangle(img, pt1 = (x0, y0), pt2 = (x0 + w, y0 + h), color = cols, thickness=2)

cv2.imshow("Image", img)
cv2.waitKey(0)