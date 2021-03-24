import torch

def matmul(x, y, axis):
    x = torch.swapaxes(x, -1, axis)
    output = torch.matmul(x, y)
    output = torch.unsqueeze(output, dim=-1)
    output = torch.swapaxes(output, -1, axis)
    output = torch.squeeze(output, dim=axis)
    return output

class Tucker(torch.nn.Module):
    def __init__(self, initial_tensor, gradient_mask, entity_embeddings, relation_embeddings):
        self.gradient_mask = gradient_mask
        self.core_tensor = torch.tensor(initial_tensor)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
    
    def forward(self, subject, relation):
        output = matmul(self.core_tensor, subject, axis=0)
        output = matmul(output, relation, axis=0)
        output = matmul(output, objects, axis=0)
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        return output
        
    def backward(self, subject, relation, y):
        relation.zero_grad()
        subject.zero_grad()
        objects.zero_grad()
        output = self.forward(subject, relation, objects)
        loss = torch.nn.BCELoss()
        loss = loss(output, y)
        loss.backward()
        self.core_tensor.grad *= self.gradient_mask
        return loss
    
    def update_grad(self, learning_rate):
        self.core_tensor = self.core_tensor - learning_rate * self.core_tensor.grad

