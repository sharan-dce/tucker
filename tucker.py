import torch
import numpy as np

def matmul(x, y, axis):
    x = torch.transpose(x, -1, axis)
    output = torch.matmul(x, y)
    output = torch.unsqueeze(output, dim=-1)
    output = torch.transpose(output, -1, axis)
    output = torch.squeeze(output, dim=axis)
    return output


def get_gradient_masked_tensor_clone(tensor, grad_mask):
    tensor_clone = tensor.detach()
    tensor = tensor * grad_mask
    tensor_clone = tensor_clone * (1.0 - grad_mask)
    return tensor + tensor_clone


class TuckER(torch.nn.Module):
    def __init__(self, num_entities, num_relations, initial_tensor, gradient_mask=None):
        if gradient_mask is None:
            gradient_mask = np.ones(initial_tensor.shape, dtype=np.float32)

        super(TuckER, self).__init__()
        assert(initial_tensor.shape == gradient_mask.shape)
        entity_embedding_dim, relation_embedding_dim = initial_tensor.shape[: 2]
        self.gradient_mask = torch.tensor(gradient_mask.astype(np.float32), requires_grad=False)
        self.core_tensor = torch.nn.Parameter(
                                        torch.tensor(initial_tensor.astype(np.float32)), 
                                        requires_grad=True
                                    )

        self.entity_embeddings = torch.nn.Embedding(num_entities, entity_embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, relation_embedding_dim)

    def forward(self, subject_index, relation_index):
        core_tensor = get_gradient_masked_tensor_clone(self.core_tensor, self.gradient_mask)
        subject = self.entity_embeddings(torch.tensor(subject_index))
        relation = self.relation_embeddings(torch.tensor(relation_index))
        objects = self.entity_embeddings.weight
        output = matmul(core_tensor, subject, axis=0)
        output = matmul(output, relation, axis=0)
        output = matmul(output, torch.transpose(objects, 0, 1), axis=0)
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        return output
