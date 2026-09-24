import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Join the arrays
print(np.concatenate((a, b)))

# Split into 3 equal parts
print(np.split(np.array([10, 20, 30, 40, 50, 60]), 3))

# Split 5 elements into 2 sections
print(np.array_split(np.array([1, 2, 3, 4, 5]), 2))