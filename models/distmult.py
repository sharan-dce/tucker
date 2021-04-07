import torch
import numpy as np
from . import tucker

class DistMult(tucker.TuckER):
    def __init__(
            self, 
            num_entities: int,
            num_relations: int,
            embedding_dim: int,
            d1=0.0, d2=0.0, d3=0.0
        ):
        self.embedding_dim = embedding_dim
        ini_tensor = np.zeros([embedding_dim, embedding_dim, embedding_dim])
        ini_tensor[np.diag_indices(embedding_dim, ndim=3)] = 1
        super(DistMult, self).__init__(
            num_entities=num_entities,
            num_relations=num_relations,
            initial_tensor=ini_tensor,
            d1=d1, d2=d2, d3=d3
        )
        self.core_tensor.requires_grad = False


if __name__ == '__main__':

    # --------------------------------------------------------
    # DisMult Model calling
    # The initial tensor does not require gradient computation
    # and all elements in the superdiagonal is 1, otherwise 0.
    # --------------------------------------------------------

    dm = DistMult(9, 9, 4)
    output = dm(torch.tensor([0, 1]), torch.tensor([5, 2]))
    print(output)

    # ini_tensor = np.zeros((4, 4, 4))  # (d_e, d_e, d_e)
    # ini_tensor[np.diag_indices(ini_tensor.shape[0], ndim=3)] = 1  # superdiagonal with 1
    # model = TuckER(9, 9, ini_tensor, np.zeros_like(ini_tensor).astype(np.float32))
    # output = model([0, 1], [5, 2])
