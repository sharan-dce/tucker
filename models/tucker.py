import torch
from typing import List
import numpy as np
from torch.nn.init import xavier_normal_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def batched_tensorvectormul(x, y, axis):
#     '''
#     x: tensor of shape b x * x d x * where * denotes any dimensions in between
#     y: matrix of shape b x d
#     '''
#     assert(len(y.shape) == 2)
#     assert(axis in range(1, len(x.shape)))
#     assert(x.shape[0] == y.shape[0])
#     assert(x.shape[axis] == y.shape[1])
#     y_new_shape = [x_dim if ax == axis or ax == 0 else 1 for ax, x_dim in enumerate(x.shape)]
#     y = torch.reshape(input=y, shape=y_new_shape)
#     broadcasted_hadamard = x * y
#     product = torch.sum(broadcasted_hadamard, axis=axis)
#     return product


def get_gradient_masked_tensor_clone(tensor, grad_mask):
    '''
    Creates a copy of 'tensor' that sends non zero gradients only at areas marked as '1' in 'grad_mask'
    '''
    tensor_clone = tensor.detach()
    tensor = tensor * grad_mask
    tensor_clone = tensor_clone * (1.0 - grad_mask)
    return tensor + tensor_clone


def tucker_multiplication(
    core, s, r, o,
    d1: torch.nn.Dropout, d2: torch.nn.Dropout, d3: torch.nn.Dropout,
    b1: torch.nn.BatchNorm1d, b2: torch.nn.BatchNorm1d):
    '''
    core of shape e x r x e
    s of shape b x e
    r of shape b x r
    o of shape n x e
    b refers to the batch size, and n the number of objects
    '''

    x = b1(s)
    x = d1(x)
    x = x.view(-1, 1, s.size(1))

    core_mat = torch.mm(r, core.view(r.size(1), -1))
    core_mat = core_mat.view(-1, s.size(1), s.size(1))
    core_mat = d2(core_mat)

    x = torch.bmm(x, core_mat)
    x = x.view(-1, s.size(1))
    x = b2(x)
    x = d3(x)
    x = torch.mm(x, o.transpose(1, 0))
    return x


class TuckER(torch.nn.Module):
    def __init__(
            self, 
            num_entities: int,
            num_relations: int,
            initial_tensor,
            gradient_mask=None,
            d1=0.0, d2=0.0, d3=0.0,
            initial_entity_embeddings=None,
            initial_relation_embeddings=None
        ):
        if gradient_mask is None:
            gradient_mask = np.ones(initial_tensor.shape, dtype=np.float32)

        super(TuckER, self).__init__()
        assert(initial_tensor.shape == gradient_mask.shape)
        entity_embedding_dim, relation_embedding_dim = initial_tensor.shape[: 2]
        self.gradient_mask = torch.tensor(gradient_mask.astype(np.float32), requires_grad=False).to(device)
        self.core_tensor = torch.nn.Parameter(
                                        torch.tensor(initial_tensor.astype(np.float32)).to(device),
                                        requires_grad=True
                                    )

        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(d) for d in [d1, d2, d3]])
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(entity_embedding_dim),
            torch.nn.BatchNorm1d(entity_embedding_dim)
        ])

        self.entity_embeddings = torch.nn.Embedding(num_entities, entity_embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        self.relation_embeddings = torch.nn.Embedding(num_relations, relation_embedding_dim)
        xavier_normal_(self.relation_embeddings.weight.data)
        if initial_entity_embeddings is not None:
            self.set_entity_embeddings(initial_entity_embeddings)
        if initial_relation_embeddings is not None:
            self.set_relation_embeddings(initial_relation_embeddings)
    
    def set_entity_embeddings(self, entity_embeddings):
        entity_embeddings = torch.from_numpy(entity_embeddings)
        self.entity_embeddings.weight.data.copy_(entity_embeddings)

    def set_relation_embeddings(self, relation_embeddings):
        relation_embeddings = torch.from_numpy(relation_embeddings)
        self.relation_embeddings.weight.data.copy_(relation_embeddings)

    def forward(self, subject_index, relation_index):
        batch_size = subject_index.shape[0]
        core_tensor = get_gradient_masked_tensor_clone(self.core_tensor, self.gradient_mask)
        subject = self.entity_embeddings(torch.LongTensor(subject_index).to(device))
        relation = self.relation_embeddings(torch.LongTensor(relation_index).to(device))
        objects = self.entity_embeddings.weight

        if len(relation.shape) == 1:
            relation = torch.unsqueeze(relation, axis=0)
        if len(subject.shape) == 1:
            subject = torch.unsqueeze(subject, axis=0)

        output = tucker_multiplication(
            core_tensor, 
            subject, 
            relation, 
            objects, 
            *self.dropouts, 
            *self.batch_norms
        )
        output = torch.sigmoid(output)
        output = torch.reshape(output, [batch_size, -1])
        return output
