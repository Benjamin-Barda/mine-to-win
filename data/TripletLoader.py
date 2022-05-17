from zmq import device
import torch
import random

class TripletLoader:
    def __init__(self, dataset, classes = 2, batch_size=32,  device = None):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.classes = classes + 1
        self.available_images = [list() for _ in range(classes + 1)]

        for i in range(len(dataset)):
            labels = dataset.data.iloc[i]
            for j, label in enumerate(labels):
                if len(label) > 0:
                    self.available_images[j + 1].append(i)
                    break
            
            self.available_images[0].append(i)

        self.ait = [torch.LongTensor(lst) for lst in self.available_images]

    def __iter__(self):
        batch_remainder = 0
        min_size = self.batch_size // self.classes
        idxs = torch.empty((3, self.batch_size), dtype=torch.int64)
        while True:
            idxs = [[0] * self.batch_size for _ in range(3)]
            n_per_class = [min_size] * self.classes
            rem = self.batch_size % self.classes
            while rem > 1:
                n_per_class[batch_remainder] += 1
                batch_remainder = (batch_remainder + 1) % self.classes
                rem -= 1

            accu = 0
            for i, lst in enumerate(self.ait):
                idx_lst = list(range(len(lst)))
                random.shuffle(idx_lst)
                idxs[0][accu : n_per_class[i] + accu] = lst[idx_lst[:n_per_class[i]]]
                idxs[1][accu : n_per_class[i] + accu] = lst[idx_lst[n_per_class[i]:2*n_per_class[i]]]
                negs = list(range(1, self.classes))
                negs = random.choices(negs, k=n_per_class[i])
                for n, k in enumerate([(j + i) % self.classes for j in negs]):
                    choice = random.choice(self.available_images[k])
                    idxs[2][accu + n] = choice
                accu += n_per_class[i]

            anchs, a_labs = self.dataset[idxs[0]]
            pos, p_labs = self.dataset[idxs[1]]
            neg, n_labs = self.dataset[idxs[2]]
            
            outTensor = torch.vstack([anchs, pos, neg], device=device) if self.device != None else torch.stack([anchs, pos, neg]) 
            yield outTensor, list(zip(a_labs, p_labs, n_labs))
