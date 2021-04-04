import torch
from typing import List
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batched_tensorvectormul(x, y, axis):
    '''
    x: tensor of shape b x * x d x * where * denotes any dimensions in between
    y: matrix of shape b x d
    '''
    assert(len(y.shape) == 2)
    assert(axis in range(1, len(x.shape)))
    assert(x.shape[0] == y.shape[0])
    assert(x.shape[axis] == y.shape[1])
    y_new_shape = [x_dim if ax == axis or ax == 0 else 1 for ax, x_dim in enumerate(x.shape)]
    y = torch.reshape(input=y, shape=y_new_shape)
    broadcasted_hadamard = x * y
    product = torch.sum(broadcasted_hadamard, axis=axis)
    return product

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
    b1: torch.nn.BatchNorm1d, b2: torch.nn.BatchNorm1d
    ):
    '''
    core of shape e x r x e
    s of shape b x e
    r of shape b x r
    o of shape n x e
    b refers to the batch size, and n the number of objects
    '''
    assert(len(s.shape) == 2 and len(r.shape) == 2 and len(o.shape) == 2)
    assert(len(core.shape) == 3)
    assert(s.shape[0] == r.shape[0])
    assert(o.shape[1] == s.shape[1])
    assert(list(core.shape) == [s.shape[1], r.shape[1], o.shape[1]])

    batch_size = s.shape[0]

    core = torch.unsqueeze(core, axis=0).repeat([batch_size, 1, 1, 1])

    s = b1(s)
    s = d1(s)
    output = batched_tensorvectormul(x=core, y=r, axis=2)
    output = d2(output)
    output = batched_tensorvectormul(x=output, y=s, axis=1)
    output = b2(output)
    output = d3(output)
    output = torch.matmul(output, torch.transpose(input=o, dim0=0, dim1=1))

    return output


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
                                        torch.tensor(initial_tensor.astype(np.float32)), 
                                        requires_grad=True
                                    )

        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(d) for d in [d1, d2, d3]])
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(entity_embedding_dim), torch.nn.BatchNorm1d(entity_embedding_dim)])

        self.entity_embeddings = torch.nn.Embedding(num_entities, entity_embedding_dim).to(device)
        self.relation_embeddings = torch.nn.Embedding(num_relations, relation_embedding_dim).to(device)
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

        e1 = self.entity_embeddings(subject_index.to(device))
        x = self.batch_norms[0](e1)
        x = self.dropouts[0](x)
        x = x.view(-1, 1, e1.size(1))

        r = self.relation_embeddings(relation_index.to(device))
        W_mat = torch.mm(r, self.core_tensor.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.dropouts[1](W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.batch_norms[1](x)
        x = self.dropouts[2](x)
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
