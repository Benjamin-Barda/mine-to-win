# Reorder the JSON st coordinates are [ul-x, ul-y, width, height]

import json

mobList = ['Creeper', 'Pig', 'purge', 'written']

directory = 'jsons/'


jsons = ["OUTPUT-pig1.json","OUTPUT-pig2.json","OUTPUT-pig3.json","OUTPUT-pig4.json",
         "OUTPUT-pig5.json","OUTPUT-pig6.json","OUTPUT-pig7.json","OUTPUT-pig8.json",
         "OUTPUT-creeper1.json","OUTPUT-creeper2.json","OUTPUT-creeper3.json","OUTPUT-creeper4.json",
         "OUTPUT-creeper5.json","OUTPUT-creeper6.json","OUTPUT-creeper7.json","OUTPUT-creeper8.json",
         "OUTPUT-null1.json","OUTPUT-null2.json","OUTPUT-null3.json","OUTPUT-null4.json",
         "OUTPUT-null5.json","OUTPUT-null6.json","OUTPUT-null7.json","OUTPUT-null8.json",
         "OUTPUT-seanull1.json"]


for json_name in jsons:
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
            
             if mob == 'purge':
                if boxes:
                    break
                else:
                    parsed[image_name][mob] = dic[image_name][mob]
                    continue
             if mob == 'written': 
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