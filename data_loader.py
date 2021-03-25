from collections import defaultdict
import torch

DATA_FOLDER = 'data'


class DataLoader:
    def __init__(self, dataset: str):
        self.data = {}

        dsets = ['train', 'valid', 'test']

        for dset in dsets:
            with open(f'{DATA_FOLDER}/{dataset}/{dset}.txt') as f:
                self.data[dset] = list(map(str.split, f.read().strip().split('\n')))

        self.entities = self._get_entities()
        self.relations = self._get_relations()
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self._determine_entity_mapping()
        self._determine_relation_mapping()

    def get_1_to_n_train_data(self):
        sr_pairs = defaultdict(list)
        ro_pairs = defaultdict(list)

        for s, r, o in self.data['train']:
            s_idx = self.entity_to_idx[s]
            r_idx = self.relation_to_idx[r]
            o_idx = self.entity_to_idx[o]
            sr_pairs[(s_idx, r_idx)].append(o_idx)
            ro_pairs[(r_idx, o_idx)].append(s_idx)

        return sr_pairs, ro_pairs

    def map_data_to_indices(self, dataset: str):
        if dataset not in ['train', 'valid', 'test']:
            raise DataLoaderException('Wrong dataset name (required one of train/valid/test)')

        mapped_data = []

        for s, r, o in self.data[dataset]:
            s_idx = self.entity_to_idx[s]
            r_idx = self.relation_to_idx[r]
            o_idx = self.entity_to_idx[o]
            mapped_data.append([s_idx, r_idx, o_idx])

        return mapped_data

    def get_embeddings(self, de: int, dr: int):
        E = torch.nn.Embedding(len(self.entities), de)
        R = torch.nn.Embedding(len(self.relations), dr)
        return E, R

    def _get_entities(self):
        entities = set()

        for _, d in self.data.items():
            entities.update(set([s for s, _, _ in d] + [o for _, _, o in d]))

        return entities

    def _get_relations(self):
        relations = set()

        for _, d in self.data.items():
            relations.update(set([r for _, r, _ in d]))

        return relations

    def _determine_entity_mapping(self):
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}

    def _determine_relation_mapping(self):
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}


class DataLoaderException(Exception):
    pass
