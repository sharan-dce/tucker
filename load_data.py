from collections import defaultdict
import numpy as np
import torch

class DataLoader:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self._load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self._load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self._load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self._get_entities(self.data)
        self.train_relations = self._get_relations(self.train_data)
        self.valid_relations = self._get_relations(self.valid_data)
        self.test_relations = self._get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.entity_idxs = {e:i for i, e in enumerate(self.entities)}
        self.relation_idxs = {r:i for i, r in enumerate(self.relations)}
        self.cuda = torch.cuda.is_available()

    def _load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def _get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def _get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
 
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_batch):
        targets = np.zeros((len(er_vocab_batch), len(self.entities)))
        for idx, pair in enumerate(er_vocab_batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(er_vocab_batch), targets
