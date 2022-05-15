import combinatorics
import numpy as np
import torch

class TripletLoader:
    def __init__(self, dataset, classes, k=10):
        self.dataset = dataset
        self.triplets_per_class = k
        self.classes = classes

        self.available_images = []

        for i in range(len(dataset)):
            labels = dataset.data.iloc[i]
            contains = set()
            empty = True
            no = False
            for j, label in enumerate(labels):
                if len(label) > 0:
                    empty = False
                    contains.add(dataset.data.columns[j])
                    if dataset.data.columns[j] not in classes:
                        no = True
                        break
            if no:
                continue

            if empty:
                contains.add("empty")
            
            self.available_images.append((i, contains))


    def __iter__(self):
        while True:
            triplets = combinatorics.create_triplets(self.available_images, self.triplets_per_class)
            triplets = np.asarray(triplets, dtype=np.int32).T
            anchs, a_labs = self.dataset[triplets[0]]
            pos, p_labs = self.dataset[triplets[1]]
            neg, n_labs = self.dataset[triplets[2]]
            yield torch.vstack([anchs, pos, neg]), list(zip(a_labs, p_labs, n_labs))
