from typing import List, Dict, Tuple, Set
from collections import defaultdict
import torch


class DataLoader:
    def __init__(self, datapath: str, add_reverses: bool = True) -> None:
        self.data = {}

        dsets = ['train', 'valid', 'test']

        for dset in dsets:
            with open(f'{datapath}/{dset}.txt') as f:
                dset_data = list(map(str.split, f.read().strip().split('\n')))

                if add_reverses:
                    dset_data += [[o, r + '_reverse', s] for s, r, o in dset_data]

                self.data[dset] = dset_data

        self.entities = self._get_entities()
        self.relations = self._get_relations()
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self._determine_entity_mapping()
        self._determine_relation_mapping()
        self.sr_pairs, self.ro_pairs = self._determine_1_to_n_train_data()

    def get_y(self, subject_idxs: List[int], relation_idxs: List[int]) -> torch.Tensor:
        '''
        For a list of subject-relation pairs (s, r) of size L, return a binary
        matrix of size L x n_e where each column represents which objects make
        a true fact with a particular (s, r) pair.

        Index-based
        '''
        assert(len(subject_idxs) == len(relation_idxs))
        result = torch.zeros((len(self.entities), len(subject_idxs)))

        for i, (si, ri) in enumerate(zip(subject_idxs, relation_idxs)):
            for v in self.sr_pairs[(si.item(), ri.item())]:
                result[v, i] = 1

        return torch.transpose(result, 0, 1)

    def get_1_to_n_train_data(self) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
        '''
        Get the training data as two dictionaries:
        - one with the pairs (s, r) as keys and a set of objects o as values
          such that facts (s, r, o) are in the dataset
        - the other with the pairs (r, o) as keys and a set of objects s as
          values such that facts (s, r, o) are in the dataset

        This function returns the indices of the entities and relations
        '''
        return self.sr_pairs, self.ro_pairs

    def get_1_to_n_valid_data(self) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
        '''
        Get the validation data as two dictionaries:
        - one with the pairs (s, r) as keys and a set of objects o as values
          such that facts (s, r, o) are in the dataset
        - the other with the pairs (r, o) as keys and a set of objects s as
          values such that facts (s, r, o) are in the dataset

        This function returns the indices of the entities and relations
        '''
        sr_pairs = defaultdict(set)
        ro_pairs = defaultdict(set)

        for s, r, o in self.data['valid']:
            s_idx = self.entity_to_idx[s]
            r_idx = self.relation_to_idx[r]
            o_idx = self.entity_to_idx[o]
            sr_pairs[(s_idx, r_idx)].add(o_idx)
            ro_pairs[(r_idx, o_idx)].add(s_idx)

        return sr_pairs, ro_pairs

    def _determine_1_to_n_train_data(self) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
        '''
        Reorganise the training data such that for all pairs (s, r) of a
        subject and a relation (respectively (r, o) of a relation and an
        object) we have a set of objects (respectively subjects) such that the
        triple (s, r, o) makes a true fact in the dataset.

        This function returns the indices of the entities and relations
        '''
        sr_pairs = defaultdict(set)
        ro_pairs = defaultdict(set)

        for s, r, o in self.data['train']:
            s_idx = self.entity_to_idx[s]
            r_idx = self.relation_to_idx[r]
            o_idx = self.entity_to_idx[o]
            sr_pairs[(s_idx, r_idx)].add(o_idx)
            ro_pairs[(r_idx, o_idx)].add(s_idx)

        return sr_pairs, ro_pairs

    def get_all_facts(self, dataset: str) -> List[Tuple[int, int, int]]:
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
            mapped_data.append((s_idx, r_idx, o_idx))

        return mapped_data

    def get_embeddings(self, de: int, dr: int) -> Tuple[torch.nn.Embedding, torch.nn.Embedding]:
        '''
        Given an entity embedding dimension `de` and the relation embedding
        dimension `dr`, return the embeddings for the entities and relations in
        the dataset
        '''
        E = torch.nn.Embedding(len(self.entities), de)
        R = torch.nn.Embedding(len(self.relations), dr)
        return E, R

    def _get_entities(self) -> Set[str]:
        '''
        Get all the entities present in any of the datasets
        '''
        entities = set()

        for _, d in self.data.items():
            entities.update(set([s for s, _, _ in d] + [o for _, _, o in d]))

        return entities

    def _get_relations(self) -> Set[str]:
        '''
        Get all the relations present in any of the datasets
        '''
        relations = set()

        for _, d in self.data.items():
            relations.update(set([r for _, r, _ in d]))

        return relations

    def _determine_entity_mapping(self) -> None:
        '''
        Map entities to indices from 0 to len(self.entities) - 1
        '''
        self.entity_to_idx = {e: i for i, e in enumerate(sorted(self.entities))}

    def _determine_relation_mapping(self) -> None:
        '''
        Map relations to indices from 0 to len(self.relations) - 1
        '''
        self.relation_to_idx = {r: i for i, r in enumerate(sorted(self.relations))}


class DataLoaderException(Exception):
    pass
