import torch
from typing import List
import numpy as np
from torch.nn.init import xavier_normal_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                                    dtype=torch.float, device=device, requires_grad=True))

        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

        self.input_dropout = torch.nn.Dropout(d1)
        self.hidden_dropout1 = torch.nn.Dropout(d2)
        self.hidden_dropout2 = torch.nn.Dropout(d3)

        self.bn0 = torch.nn.BatchNorm1d(dim1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)
    

    def forward(self, subject_index, relation_index):
        e1_idx = subject_index
        r_idx = relation_index
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
