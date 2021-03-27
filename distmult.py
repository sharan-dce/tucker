import torch
import numpy as np
from tucker import matmul


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations, initial_tensor, gradient_mask=None):
        """

        :param num_entities:    n_e
        :param num_relations:   n_r
        :param initial_tensor:  (d_e, d_e, d_e) only shape matters
        :param gradient_mask:   (d_e, d_e, d_e)
        """
        if gradient_mask is None:
            gradient_mask = np.ones(initial_tensor.shape, dtype=np.float32)

        super(DistMult, self).__init__()
        assert(initial_tensor.shape == gradient_mask.shape)
        assert(initial_tensor.shape[0] == initial_tensor.shape[1])
        assert (initial_tensor.shape[1] == initial_tensor.shape[2])
        entity_embedding_dim = initial_tensor.shape[0]
        self.gradient_mask = torch.tensor(gradient_mask.astype(np.float32), requires_grad=False)

        # The core tensor of DistMult is a static tensor so it does not require gradients.
        # Z_pqr = 1 if p == q == r else 0
        core_tensor = np.zeros((initial_tensor.shape[0], initial_tensor.shape[1], initial_tensor.shape[2]))
        core_tensor[np.diag_indices(initial_tensor.shape[0], ndim=3)] = 1
        self.core_tensor = torch.nn.Parameter(
                                        torch.tensor(core_tensor.astype(np.float32)),
                                        requires_grad=False
                                    )

        self.entity_embeddings = torch.nn.Embedding(num_entities, entity_embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, entity_embedding_dim)  # (n_r, d_e)

    def forward(self, subject_index, relation_index):
        subject = self.entity_embeddings(torch.tensor(subject_index))      # (d_e)
        relation = self.relation_embeddings(torch.tensor(relation_index))  # (d_e)
        objects = self.entity_embeddings.weight                            # (n_e, d_e)
        output = matmul(self.core_tensor, subject, axis=0)                 # (d_e, d_e, d_e) *0 (d_e) = (d_e, d_e)
        output = matmul(output, relation, axis=0)                          # (d_e, d_e) *0 (d_e) = (d_e)
        output = matmul(output, torch.transpose(objects, 0, 1), axis=0)    # (d_e) *0 (d_e, n_e) = (n_e)
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        return output


if __name__ == '__main__':
    ini = np.ones((4, 4, 4))
    model = DistMult(7, 8, ini.astype(np.float32))
    output = model(0, 1)
