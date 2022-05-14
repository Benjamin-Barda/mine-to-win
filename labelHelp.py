import numpy as np
import cv2
import os
import json
import sys


imageFolderPath = "imgs/"

MOB_MAP  =  {
    ord('1') : ('ZOMBIE', (50,120,20)),
    ord('2') : ('SCHELETRO', (225,225,225)),
    ord('3') : ('MAIALE', (205,170,230)),
    ord('4') : ('MUCCA', (50,70,120)),
    ord('5') : ('GALLINA', (5,5,200))
}

H, W = (300,150)

white_background = np.zeros((H,W,3))


i = 0
mult = 20
 
for key, value in MOB_MAP.items():
    i+=1
    cv2.putText(white_background, chr(key) + ": " + value[0], (10,i*mult), cv2.FONT_HERSHEY_PLAIN, 1, (value[1][0] / 255,value[1][1] / 255,value[1][2] / 255))


DUMP = {}


def draw(event, x, y, flags, params): 

    global startX, startY, isDrawing, endX, endY

    if event == cv2.EVENT_LBUTTONDOWN:
        isDrawing = True
        startX = x 
        startY = y 
    
    if event == cv2.EVENT_LBUTTONUP : 
        if isDrawing :
            endX = x
            endY = y

            rectList.append((startX, startY, endX, endY, col))
            isDrawing = False
            DUMP[image][selectedLabel].append((startX, startY, endX, endY))

    if event == cv2.EVENT_RBUTTONDOWN : 
        if len(rectList) > 0 : 
            rectList.pop(-1)
            DUMP[image][selectedLabel].pop(-1)
        

for image in os.listdir(imageFolderPath):

    cv2.namedWindow(image)
    cv2.setMouseCallback(image, draw)

    cv2.namedWindow('LEGEND')
    cv2.imshow('LEGEND', white_background)

    pathname = imageFolderPath + image

    img = cv2.imread(pathname)
    DUMP[image] = {x[0] : [] for x in MOB_MAP.values()}

    run = True

    selectedLabel, col = MOB_MAP[ord('1')]
    rectList = []

    while run :
        
        buff = img.copy()        

        for rect in rectList : 
            x0, y0, x1, y1, cols = rect
            cv2.rectangle(buff, pt1 = (x0, y0), pt2 = (x1, y1), color = cols, thickness=2)


        cv2.imshow(image, buff)
        
        key = cv2.waitKey(10)

        # Change selection
        if key in MOB_MAP.keys() : 
            selectedLabel = MOB_MAP[key][0]
            col = MOB_MAP[key][1]

        # Go to the next image
        elif key == ord('0') :
            # Update json file 
            run = False
            cv2.destroyWindow(image)
        
        elif key == ord('q') : 
            with open('OUTPUT.txt', 'w') as out : 
                out.write(json.dumps(DUMP))
                out.close()
            sys.exit()



with open('OUTPUT.json', 'w') as out : 
    out.write(json.dumps(DUMP, sort_keys=True, indent=4))
    out.close()