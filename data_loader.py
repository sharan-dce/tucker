from typing import List

class DataLoader:

    def __init__(
        self,
        datapath: str,
        add_reverses: bool=True
        ):
        self.data = {}

        dsets = ['train', 'valid', 'test']

        for dset in dsets:
            with open(f'{datapath}/{dset}.txt') as f:
                dset_data = list(map(str.split, f.read().strip().split('\n')))

                if add_reverses:
                    dset_data += [[o, r + '_reverse', s] for s, r, o in dset_data]

                self.data[dset] = dset_data


        self.data = dsets['train'] + dsets['test'] + dsets['valid']
        self.entities = self._get_entities()
        self.relations = self._get_relations()

    def _get_entities(self) -> List[str]:
        '''
        Get all the entities present in any of the datasets
        '''
        entities = set()

        for _, d in self.data.items():
            entities.update(set([s for s, _, _ in d] + [o for _, _, o in d]))

        return list(entities)

    def _get_relations(self) -> List[str]:
        '''
        Get all the relations present in any of the datasets
        '''
        relations = set()

        for _, d in self.data.items():
            relations.update(set([r for _, r, _ in d]))

        return list(relations)

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
