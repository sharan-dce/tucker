import numpy as np
import torch
from torch.nn.init import xavier_normal_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TuckER(torch.nn.Module):
    def __init__(self,
        num_entities: int,
        num_relations: int,
        initial_tensor: np.ndarray,             # pass in a rdim x edim x edim tensor to initialize with
        d1: float=0.0, d2: float=0.0, d3: float=0.0,     # 3 dropout values
        ):
        super(TuckER, self).__init__()
        assert(initial_tensor.shape[1] == initial_tensor.shape[2])

        self.rdim, self.edim = initial_tensor.shape[: 2]

        # Get parameters for model
        self.entity_embeddings = torch.nn.Embedding(num_entities, self.edim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, self.rdim)
        self.core_tensor = torch.nn.Parameter(torch.tensor(
                                    initial_tensor.astype(np.float32),
                                    device=device,
                                    requires_grad=True
                                ))
        # get batch norms and dropouts
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim) for dim in [self.edim, self.edim]])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(drop_prob) for drop_prob in [d1, d2, d3]])
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
    
    def _process_entities(self, entities):
        # batch norm first on entity embeddings
        entities = self.batch_norms[0](entities)
        entities = self.dropouts[0](entities).unsqueeze(1)
        return entities
    
    def _core_relations_prod(self, relations):
        # first product - alond axis 0 of core tensor - relation dimension
        core_view = self.core_tensor.view(self.rdim, -1)
        print(relations.dtype, core_view.dtype)
        core = torch.mm(relations, core_view)
        # multiplication done-
        # now reshape core tensor back
        # first dimension must have vanished
        # batch size will be first after this operation
        core = core.view(-1, self.edim, self.edim)
        core = self.dropouts[1](core)
        return core

    def _core_entities_product(self, core, entities):
        batch_size = entities.size(0)
        output = torch.bmm(entities, core).view(batch_size, -1)
        # now have batch_size x matrices - 
        # batch multiply them
        # and flatten along final dimension
        output = self.batch_norms[1](output)
        output = self.dropouts[2](output)
        # dot with all entities to get scalars
        all_entities = self.entity_embeddings.weight
        output = torch.mm(all_entities, output.transpose(0, 1))
        return output


    def forward(self, 
        batched_entities: torch.tensor, 
        batched_relations: torch.tensor
        ):
        assert(batched_entities.shape == batched_relations.shape)
        batch_size = batched_entities.size(0)

        entities = self.entity_embeddings(batched_entities)
        relations = self.relation_embeddings(batched_relations)

        entities = self._process_entities(entities=entities)
        # core will lose one dimensions after this
        multiplied_core = self._core_relations_prod(relations=relations)
        # core will lose other 2 dimensions after this to
        # return a scalar
        output = self._core_entities_product(multiplied_core, entities)
        
        probs = torch.sigmoid(output)
        return probs.transpose(0, 1)
