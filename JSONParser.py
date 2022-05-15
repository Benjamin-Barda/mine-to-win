# Reorder the JSON st coordinates are [ul-x, ul-y, width, height]

import json

mobList = ['Chicken', 'Cow', 'Creeper', 'Pig', 'Sheep', 'Skeleton', 'Slime', 'Spider', 'Wolf', 'Zombie', 'purge', 'written']

directory = 'jsons/'

json_name = 'OUTPUT-world3.json' 

path_to_json = directory + json_name

with open(path_to_json, 'r') as f : 
    dic = json.loads(f.read())
    f.close()

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

            mid_x = (x0 + x1) // 2
            mid_y = (y0 + y1) // 2

            parsed[image_name][mob].append([mid_x, mid_y, W, H])
        
            
with open(directory + 'PARSED' + json_name, 'w') as g:
    g.write(json.dumps(parsed))
    g.close()