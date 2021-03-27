import torch
from tucker import TuckER
import numpy as np

t = TuckER(9, 9, np.random.normal(size=[3, 11, 3]).astype(np.float32), np.ones([3, 11, 3]).astype(np.float32))
output = t([0, 1], [5, 2])

# output = output.reshape([-1])
# loss = torch.nn.BCELoss()(output, y)
# loss.backward()

print(output)