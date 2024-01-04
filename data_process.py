import os
import numpy as np
import torch
from collections import defaultdict


class Dataset:
    def __init__(self, data_dir, arity_lst, device):
        self.data_dir = data_dir
        self.ent2id, self.rel2id = self.str2id(self.data_dir)
        self.device = device
        self.arity_lst = arity_lst
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.load_data(self.data_dir, arity_lst)
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)
        self.batch_index = 0



    def load_data(self, data_dir, arity_lst):
        print("Loading {} Dataset".format(data_dir.split("/")[-1]))
        self.train = self.read_tuples(os.path.join(data_dir, "train.txt"), arity_lst, "train")
        self.valid = self.read_tuples(os.path.join(data_dir, "valid.txt"), arity_lst, "valid")
        self.test = self.read_tuples(os.path.join(data_dir, "test.txt"), arity_lst, "test")
        self.all_test = []
        self.all_valid = []
        for arity in arity_lst:
            for fact in self.valid[arity]:
                self.all_valid.append(fact)
            for fact in self.test[arity]:
                self.all_test.append(fact)


    def read_tuples(self, dataset, arity_lst, mode):
        if mode == "train":
            self.inc = defaultdict(list)
        data = {}
        for arity in arity_lst:
            data[arity] = []
        with open(dataset) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                rel = self.rel2id[line[0]]
                ents = [self.ent2id[i] for i in line[1:]]
                arity = len(ents)
                if arity in data:
                    data[arity].append(np.array([rel]+ents))
        return data



    def str2id(self, path):
        ent2id, rel2id = {"": 0}, {"": 0}
        with open(os.path.join(path, "entities.dict")) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                id, ent = line[0], line[1]
                if ent not in ent2id:
                    ent2id[ent] = int(id)+1
            f.close()

        with open(os.path.join(path, "relations.dict")) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                id, rel = line[0], line[1]
                if rel not in rel2id:
                    rel2id[rel] = int(id)+1
            f.close()

        return ent2id, rel2id

    def next_pos_batch(self, batch_size, arity):
        if self.batch_index + batch_size < len(self.train[arity]):
            batch = self.train[arity][self.batch_index: self.batch_index + batch_size]
            self.batch_index += batch_size
        elif self.batch_index + batch_size >= len(self.train[arity]):
            batch = self.train[arity][self.batch_index:]
            self.batch_index = 0
        batch = np.append(batch, -np.ones((len(batch), 1)), axis=1).astype("int")  # appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int")  # appending the 0 arity
        return batch

    def next_batch(self, batch_size, neg_ratio, arity, device):
        pos_batch = self.next_pos_batch(batch_size, arity)
        batch = self.generate_neg(pos_batch, neg_ratio, arity)
        batch = torch.tensor(batch).long().to(device)

        return batch

    def generate_neg(self, pos_batch, neg_ratio, arity):
        arities = [arity + 1 - (t == 0).sum() for t in pos_batch[:, 1:]]
        pos_batch[:, -1] = arities
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio) for i, c in enumerate(pos_batch)], axis=0)
        return neg_batch

    def neg_each(self, arr, arity, nr):
        arr[0,-2] = 1
        for a in range(arity):
            arr[a* nr + 1:(a + 1) * nr + 1, a + 1] = np.random.randint(low=1, high=self.num_ent, size=nr)
        return arr




    def is_last_batch(self):
        return (self.batch_index == 0)

