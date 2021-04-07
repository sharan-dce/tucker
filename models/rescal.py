import torch
import numpy as np
from . import tucker


# RESCAL model supporting batches of subjects and relations

class RESCAL(tucker.TuckER):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            d1=0.0, d2=0.0, d3=0.0
        ):
        super(RESCAL, self).__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            initial_tensor=np.random.normal(size=[num_relations, embedding_dim, embedding_dim]),
            initial_relation_embeddings=np.identity(num_relations, dtype=np.float32),
            d1=d1, d2=d2, d3=d3
        )
        # self.core_tensor.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False

if __name__ == '__main__':
    model = RESCAL(num_entities=9, num_relations=7, embedding_dim=3)
    # model = RESCAL(7, 8, np.random.normal(size=[3, 8, 3]).astype(np.float32))
    output = model(torch.tensor([0, 1]), torch.tensor([2, 3]))
    print(output.shape)
