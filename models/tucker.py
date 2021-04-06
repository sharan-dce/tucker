import torch
from typing import List
import numpy as np
from torch.nn.init import xavier_normal_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            d1=0.0, d2=0.0, d3=0.0,
        ):

        super(TuckER, self).__init__()
        
        dim1, dim2 = initial_tensor.shape[: 2]
        self.E = torch.nn.Embedding(num_entities, dim1)
        self.R = torch.nn.Embedding(num_relations, dim2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim2, dim1, dim1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(d1)
        self.hidden_dropout1 = torch.nn.Dropout(d2)
        self.hidden_dropout2 = torch.nn.Dropout(d3)

        self.bn0 = torch.nn.BatchNorm1d(dim1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)
    

    def forward(self, subject_index, relation_index):
        e1_idx = subject_index.to(device)
        r_idx = relation_index.to(device)
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
