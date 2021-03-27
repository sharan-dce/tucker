import torch
from tucker import TuckER
import numpy as np

t = TuckER(9, 9, np.random.normal(size=[3, 11, 3]).astype(np.float32), np.ones([3, 11, 3]).astype(np.float32))
output = t([0], [5])

y = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float32)

output = output.reshape([-1])
loss = torch.nn.BCELoss()(output, y)
loss.backward()

print(t.core_tensor.grad.shape)