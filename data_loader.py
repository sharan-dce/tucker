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
        self.sr_pairs, self.ro_pairs = self._determine_1_to_n_train_data()

    def get_y(self, subject_idxs, relation_idxs):
        '''
        For a list of subject-relation pairs (s, r) of size L, return a binary
        matrix of size n_e x L where each column represents which objects make
        a true fact with a particular (s, r) pair.

        Index-based
        '''
        assert(len(subject_idxs) == len(relation_idxs))
        result = torch.zeros((len(self.entities), len(subject_idxs)))

        for i, (si, ri) in enumerate(zip(subject_idxs, relation_idxs)):
            for v in self.sr_pairs[(si, ri)]:
                result[v, i] = 1

        return result

    def get_1_to_n_train_data(self):
        '''
        Get the training data as two dictionaries:
        - one with the pairs (s, r) as keys and a list of objects o as values
          such that facts (s, r, o) are in the dataset
        - the other with the pairs (r, o) as keys and a list of objects s as
          values such that facts (s, r, o) are in the dataset
        
        This function returns the indices of the entities and relations
        '''
        return self.sr_pairs, self.ro_pairs

    def _determine_1_to_n_train_data(self):
        '''
        Reorganise the training data such that for all pairs (s, r) of a
        subject and a relation (respectively (r, o) of a relation and an
        object) we have a list of objects (respectively subjects) such that the
        triple (s, r, o) makes a true fact in the dataset.

        This function returns the indices of the entities and relations
        '''
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
        '''
        Given a dataset (one of train/valid/test), return all the facts (s,
        r, o) present in the dataset.

        This function returns the indices of the entities and relations
        '''
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
        '''
        Given an entity embedding dimension `de` and the relation embedding
        dimension `dr`, return the embeddings for the entities and relations in
        the dataset
        '''
        E = torch.nn.Embedding(len(self.entities), de)
        R = torch.nn.Embedding(len(self.relations), dr)
        return E, R

    def _get_entities(self):
        '''
        Get all the entities present in any of the datasets
        '''
        entities = set()

        for _, d in self.data.items():
            entities.update(set([s for s, _, _ in d] + [o for _, _, o in d]))

        return entities

    def _get_relations(self):
        '''
        Get all the relations present in any of the datasets
        '''
        relations = set()

        for _, d in self.data.items():
            relations.update(set([r for _, r, _ in d]))

        return relations

    def _determine_entity_mapping(self):
        '''
        Map entities to indices from 1 to len(self.entities) - 1
        '''
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}

    def _determine_relation_mapping(self):
        '''
        Map relations to indices from 1 to len(self.relations) - 1
        '''
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}


class DataLoaderException(Exception):
    pass
