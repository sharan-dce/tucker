import torch
import numpy as np
from tucker import get_gradient_masked_tensor_clone


# RESCAL model supporting batches of subjects and relations
class RESCAL(torch.nn.Module):
    def __init__(self, num_entities, num_relations, initial_tensor, gradient_mask=None):
        """

        :param num_entities:    n_e
        :param num_relations:   n_r
        :param initial_tensor:  (d_e, n_r, d_e)
        :param gradient_mask:   (d_e, n_r, d_e)
        """
        if gradient_mask is None:
            gradient_mask = np.ones(initial_tensor.shape, dtype=np.float32)

        super(RESCAL, self).__init__()
        assert(initial_tensor.shape == gradient_mask.shape)
        assert(num_relations == initial_tensor.shape[1])
        entity_embedding_dim = initial_tensor.shape[0]
        self.gradient_mask = torch.tensor(gradient_mask.astype(np.float32), requires_grad=False)
        self.core_tensor = torch.nn.Parameter(
                                        torch.tensor(initial_tensor.astype(np.float32)),
                                        requires_grad=True
                                    )

        self.entity_embeddings = torch.nn.Embedding(num_entities, entity_embedding_dim)

    def forward(self, subject_index, relation_index):
        core_tensor = get_gradient_masked_tensor_clone(self.core_tensor, self.gradient_mask)  # (d_e, n_r, d_e)
        relation_matrix = core_tensor[:, relation_index, :].permute(1, 0, 2)                  # (bs, d_e, d_e)
        subject = self.entity_embeddings(torch.tensor(subject_index))                         # (bs, d_e)
        objects = self.entity_embeddings.weight                                               # (n_e, d_e)

        output = torch.bmm(subject.unsqueeze(1), relation_matrix)       # (bs, 1, d_e) * (bs, d_e, d_e) = (bs, 1, d_e)
        expanded_objects = torch.transpose(objects, 0, 1).expand(len(subject_index), -1, -1)
        output = torch.matmul(output, expanded_objects)                 # (bs, 1, d_e) * (bs, d_e, n_e) = (bs, 1, n_e)
        output = output.squeeze()                                       # (bs, 1, n_e) => (bs, n_e)
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        return output


if __name__ == '__main__':
    model = RESCAL(7, 8, np.random.normal(size=[3, 8, 3]).astype(np.float32))
    output = model([0, 1], [0, 1])
