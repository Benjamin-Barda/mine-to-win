# Reorder the JSON st coordinates are [ul-x, ul-y, width, height]

import json

mobList = ['Creeper', 'Pig', "Zombie", "Sheep", 'purge', 'written']

directory = 'jsons/'


jsons = ["OUTPUT-pig1.json","OUTPUT-pig2.json","OUTPUT-pig3.json","OUTPUT-pig4.json", "OUTPUT-pig5.json",
         "OUTPUT-pig6.json","OUTPUT-pig7.json","OUTPUT-pig8.json","OUTPUT-pig9.json","OUTPUT-pig10.json",
         "OUTPUT-creeper1.json","OUTPUT-creeper2.json","OUTPUT-creeper3.json","OUTPUT-creeper4.json","OUTPUT-creeper5.json",
         "OUTPUT-creeper6.json","OUTPUT-creeper7.json","OUTPUT-creeper8.json","OUTPUT-creeper9.json","OUTPUT-creeper10.json",
         "OUTPUT-null1.json","OUTPUT-null2.json","OUTPUT-null3.json","OUTPUT-null4.json","OUTPUT-null5.json",
         "OUTPUT-null6.json","OUTPUT-null7.json","OUTPUT-null8.json","OUTPUT-null9.json","OUTPUT-null10.json",
         "OUTPUT-null11.json","OUTPUT-null12.json","OUTPUT-null13.json","OUTPUT-null14.json","OUTPUT-null15.json",
         "OUTPUT-null16.json","OUTPUT-null17.json","OUTPUT-null18.json","OUTPUT-null19.json","OUTPUT-seanull1.json",
         "OUTPUT-zombie1.json","OUTPUT-zombie2.json","OUTPUT-zombie3.json","OUTPUT-zombie4.json", "OUTPUT-zombie5.json",
         "OUTPUT-zombie6.json","OUTPUT-zombie7.json","OUTPUT-zombie8.json","OUTPUT-zombie9.json",
         "OUTPUT-sheep1.json","OUTPUT-sheep2.json","OUTPUT-sheep3.json","OUTPUT-sheep4.json", "OUTPUT-sheep5.json",
         "OUTPUT-sheep6.json","OUTPUT-sheep7.json","OUTPUT-sheep8.json","OUTPUT-sheep9.json",
         "OUTPUT-test1.json","OUTPUT-test2.json",
        ]


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