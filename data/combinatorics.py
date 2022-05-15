import random

# classes = ['Empty','Pig','Cow','Chicken','Sheep','Zombie','Skeleton','Creeper','Spider', 'Wolf', 'Slime']

def create_triplets(lst, triplets_per_class=300):
    # Get labels
    known_values = dict()

    for idx, img_info in lst:
        for val in img_info:
            if val in known_values:
                known_values[val].add(idx)
            else:
                known_values[val] = {idx}

    # Make as list
    for key in known_values:
        known_values[key] = list(known_values[key])

    triplets_indices = list()
    
    for key, lst in known_values.items():
        for _ in range(triplets_per_class):
            contained_stuff = set()
            # Choose anchor
            anchor = lst[random.randrange(0, len(lst) - 1)]

            for key2 in known_values:
                if anchor in known_values[key2]:
                    contained_stuff.add(key2)

            # Choose positive
            pos = anchor
            while pos == anchor:
                pos = lst[random.randrange(0, len(lst) - 1)]


            # Choose negative
            contained_negs = contained_stuff
            neg = anchor
            while not contained_negs.isdisjoint(contained_stuff):
                contained_negs = set()
                keys = list(known_values.keys())
                chosen_key = key
                while chosen_key == key:
                    chosen_key = keys[random.randrange(0, len(keys) - 1)]
                
                neg = known_values[chosen_key][random.randrange(0, len(known_values[chosen_key]) - 1)]

                for key2 in known_values:
                    if neg in known_values[key2]:
                        contained_negs.add(key2)
            
            triplets_indices.append((anchor,pos,neg))
            
    random.shuffle(triplets_indices)
    return triplets_indices


# random.seed(42)

# images = list()

# for j in range(1000):
#     n = random.randrange(0, 6)
#     lst = set()
#     for i in range(n):
#         lst.add(classes[random.randrange(0, len(classes) - 1)])

#     images.append((j,lst))

# print(create_triplets(images))
