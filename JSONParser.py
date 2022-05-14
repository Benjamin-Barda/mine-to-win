# Reorder the JSON st coordinates are [ul-x, ul-y, width, height]

import json

mobList = ['Chicken', 'Cow', 'Creeper', 'Pig', 'Sheep', 'Skeleton', 'Slime', 'Spider', 'Wolf', 'Zombie', 'purge', 'written']

path_to_json = '' 

with open(path_to_json, 'r') as f : 
    dic = json.loads(f.read())

parsed = {}

# K = Image name 
# v = {Label : boxes}
for image_name , d in dic.items() :

    parsed[image_name] = {x : [] for x in mobList}

    for mob, boxes in d.items() : 

        if mob == 'purge' or mob == 'written' : 
            parsed[image_name][mob] = dic[image_name][mob]
            continue

        for box in boxes : 

            x0, x1 = box[0], box[2]
            y0, y1 = box[1], box[3]

            W = abs(x0 - x1)
            H = abs(y0 - y1)

            upper_left_x = min(x0, x1)
            upper_left_y = min(y0, y1)

            parsed[image_name][mob].append([upper_left_x, upper_left_y, W, H])
        
            


         
