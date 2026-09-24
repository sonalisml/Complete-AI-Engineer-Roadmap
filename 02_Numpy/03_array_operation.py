import numpy as np
a = np.array([45,60,75,90,30])
print(np.sum(a))
print(np.average(a))
print(np.max(a))
print(np.min(a))
marks = np.array([
    [80, 90],
    [70, 85],
    [60, 95]
])
average = np.average(marks, axis=1)
print(average)

#reshape #flatten

aa = np.arange(1, 13)

# Reshape into 3 rows and 4 columns
bb = aa.reshape(3, 4)
print(bb)

# Convert b into a 1D array
cc = bb.flatten()
print(cc)

# Reshape into 2 rows, allowing NumPy to infer columns
dd = aa.reshape(2, -1)
print(dd)