import numpy as np

a = np.array([45, 20, 45, 10, 30, 20, 50])

# 1. Print the array in ascending order
print(np.sort(a))
# 2. Print the unique values
print(np.unique(a))
# 3. Find the indices of values greater than 30
print(np.where(a>30))
# 4. Replace values below 30 with 0 using np.where()
print(np.where(a<30, 0, a))

