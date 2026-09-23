#Convert a numpy array to pytorch tensor

import numpy as np
import torch

arr = np.array([2, 4, 6])

tensor = torch.from_numpy(arr)
print(tensor)
print(type(tensor))

arr_back = tensor.numpy()
print(arr_back)