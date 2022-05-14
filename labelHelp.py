import numpy as np
import cv2
import os
import json
import sys
import pathlib

folder = "world2"

FROM = 0
TO = 200


imageFolderPath = "imgs/"+ folder + "/"

MOB_MAP  =  {
    ord('1') : ('Pig', (205, 170, 230)),
    ord('2') : ('Cow', (50, 70, 120)),
    ord('3') : ('Chicken', (5, 5, 200)),
    ord('4') : ('Sheep', (125, 125, 121)),
    ord('5') : ('Zombie', (50, 120, 20)),
    ord('6') : ('Skeleton', (225, 225, 225)),
    ord('7') : ('Creeper', (50, 168, 58)),
    ord('8') : ('Spider', (10, 8, 13)),
    ord('9') : ('Wolf', (255, 119, 82)),
    ord('0') : ('Slime', (85, 255, 82)),
}

H, W = (300,150)

legend = np.zeros((H,W,3))

i = 0
mult = 20
 
for key, value in MOB_MAP.items():
    i+=1
    cv2.putText(legend, chr(key) + ": " + value[0], (10,i*mult), cv2.FONT_HERSHEY_PLAIN, 1, (value[1][0] / 255,value[1][1] / 255,value[1][2] / 255))


DUMP = {}

currX, currY = 0,0
mode = 0
mode_state = 0
strdX, strdY = 0,0


def draw(event, x, y, flags, params): 
    global currX, currY, strdX, strdY, mode, mode_state

    currX = x
    currY = y

    if event == cv2.EVENT_RBUTTONDOWN:
        if mode_state == 1:
            mode_state = 0
        elif len(rectList) > 0:
            rectList.pop(-1)
            DUMP[image_name][selectedLabel].pop(-1)

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if mode == 0:
        if mode_state == 0:
            strdX = x 
            strdY = y
            mode_state = 1
        else:
            rectList.append((strdX, strdY, x, y, col))
            DUMP[image_name][selectedLabel].append((strdX, strdY, x, y))
            DUMP[image_name]["written"] = True
            mode_state = 0

cv2.namedWindow('LEGEND')
cv2.imshow('LEGEND', legend)

i = FROM
all_imgs = os.listdir(imageFolderPath)
image_name = all_imgs[i]

output_name = "OUTPUT-" + folder + ".json"

output_file = pathlib.Path(output_name)

if output_file.exists():
    with open(output_name, 'r') as j:
        DUMP = json.loads(j.read())

for key, value in DUMP.items():
    if image_name != key:
        break
    elif not value.get("written", False):
        break
    i += 1
    image_name = all_imgs[i]

cv2.namedWindow("win_id")
cv2.setMouseCallback("win_id", draw)
cv2.setWindowTitle("win_id", image_name)

pathname = imageFolderPath + image_name

img = cv2.imread(pathname)
DUMP[image_name] = {x[0] : [] for x in MOB_MAP.values()}
DUMP[image_name]["written"] = False
DUMP[image_name]["purge"] = False

selectedLabel, col = MOB_MAP[ord('1')]
rectList = []

run = True

while run:
    buff = img.copy()        

    for rect in rectList: 
        x0, y0, x1, y1, cols = rect
        cv2.rectangle(buff, pt1 = (x0, y0), pt2 = (x1, y1), color = cols, thickness=2)

    if mode_state == 1:
        cv2.rectangle(buff, pt1 = (strdX, strdY), pt2 = (currX, currY), color = col, thickness=2)


    cv2.imshow("win_id", buff)
    
    key = cv2.waitKey(10)

    # Change selection
    if key in MOB_MAP.keys(): 
        selectedLabel = MOB_MAP[key][0]
        col = MOB_MAP[key][1]

    # Go to the next image
    elif key == ord('f') or key == ord('d') or key == ord('b'):
        # Update json file
        if key == ord('d'):
            DUMP[image_name]["purge"] = True
        else:
            DUMP[image_name]["purge"] = False
        if key == ord('b'):
            i -= 1
            if i < 0:
                i = 0
        else:
            i += 1
        
        if i >= len(all_imgs) or TO > 0 and i >= TO:
            run = False
            break
        
        image_name = all_imgs[i]
        cv2.setWindowTitle("win_id", image_name)
        pathname = imageFolderPath + image_name
        img = cv2.imread(pathname)
        selectedLabel, col = MOB_MAP[ord('1')]
        rectList.clear()
        if image_name not in DUMP:
            DUMP[image_name] = {x[0] : [] for x in MOB_MAP.values()}
            DUMP[image_name]["written"] = False
            DUMP[image_name]["purge"] = False
        else:
            my_dict = DUMP[image_name]
            for label, color in MOB_MAP.values():
                for vals in my_dict[label]:
                    rectList.append((vals[0],vals[1], vals[2], vals[3], color))

    elif key == ord('q'):
        run = False


with open(output_name, 'w') as out: 
    out.write(json.dumps(DUMP, sort_keys=True, indent=4))
    out.close()