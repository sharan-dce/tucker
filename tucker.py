import torch
import numpy as np

def matmul(x, y, axis):
    x = torch.transpose(x, -1, axis)
    output = torch.matmul(x, y)
    output = torch.unsqueeze(output, dim=-1)
    output = torch.transpose(output, -1, axis)
    output = torch.squeeze(output, dim=axis)
    return output

# x: tensor of shape b x * x d x * where * denotes any dimensions in between
# y: matrix of shape b x d
def batched_tensorvectormul(x, y, axis):
    assert(len(y.shape) == 2)
    assert(axis in range(1, len(x.shape)))
    assert(x.shape[0] == y.shape[0])
    assert(x.shape[axis] == y.shape[1])
    y_new_shape = [x_dim if ax == axis or ax == 0 else 1 for ax, x_dim in enumerate(x.shape)]
    y = torch.reshape(input=y, shape=y_new_shape)
    broadcasted_hadamard = x * y
    product = torch.sum(broadcasted_hadamard, axis=axis)
    return product

# creates a copy of 'tensor' that sends non zero gradients only at areas marked as '1' in 'grad_mask'
def get_gradient_masked_tensor_clone(tensor, grad_mask):
    tensor_clone = tensor.detach()
    tensor = tensor * grad_mask
    tensor_clone = tensor_clone * (1.0 - grad_mask)
    return tensor + tensor_clone


# core of shape e x r x e
# s of shape b x e
# r of shape b x r
# o of shape n x e
# b refers to the batch size, and n the number of objects
def tucker_multiplication(core, s, r, o):
    assert(len(s.shape) == 2 and len(r.shape) == 2 and len(o.shape) == 2)
    assert(len(core.shape) == 3)
    assert(s.shape[0] == r.shape[0])
    assert(o.shape[1] == s.shape[1])
    assert(list(core.shape) == [s.shape[1], r.shape[1], o.shape[1]])

    batch_size = s.shape[0]

    core = torch.unsqueeze(core, axis=0).repeat([batch_size, 1, 1, 1])
    output = batched_tensorvectormul(x=core, y=s, axis=1)
    output = batched_tensorvectormul(x=output, y=r, axis=1)
    output = torch.matmul(output, torch.transpose(input=o, dim0=0, dim1=1))

    return output


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

        if len(relation.shape) == 1:
            relation = torch.unsqueeze(relation, axis=0)
        if len(subject.shape) == 1:
            subject = torch.unsqueeze(subject, axis=0)

        output = tucker_multiplication(core_tensor, subject, relation, objects)
        print(output.shape)
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        return output
