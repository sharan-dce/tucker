import torch
import numpy as np
from tucker import TuckER


if __name__ == '__main__':

    # --------------------------------------------------------
    # DisMult Model calling
    # The initial tensor does not require gradient computation
    # and all elements in the superdiagonal is 1, otherwise 0.
    # --------------------------------------------------------

    ini_tensor = np.zeros((4, 4, 4))  # (d_e, d_e, d_e)
    ini_tensor[np.diag_indices(ini_tensor.shape[0], ndim=3)] = 1  # superdiagonal with 1
    model = TuckER(9, 9, ini_tensor, np.zeros_like(ini_tensor).astype(np.float32))
    output = model([0, 1], [5, 2])
