import numpy as np

# 1. Create matrix A
A = np.array([[2, 4],
              [6, 8]])

# 2. Print transpose of A
print("Transpose of A:")
print(A.T)

# 3. Create matrix B
B = np.array([[1, 3],
              [5, 7]])

# 4. Matrix multiplication
print("Matrix multiplication (A @ B):")
print(A @ B)

# 5. Elementwise multiplication
print("Elementwise multiplication (A * B):")
print(A * B)