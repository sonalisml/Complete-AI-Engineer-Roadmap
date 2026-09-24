import numpy as np

data = np.array([12, 18, np.nan, 24, np.nan, 36])

# 1. Identify missing values
print("Missing values:", np.isnan(data))

# 2. Count missing values
print("Missing count:", np.isnan(data).sum())

# 3. Calculate mean, ignoring NaN
print("Mean:", np.nanmean(data))

# 4. Replace NaN with -1 in a new array
data_clean = np.nan_to_num(data, nan=-1)

print("Cleaned array:", data_clean)