import numpy as np

# 1. Create a random number generator with seed 7
rng = np.random.default_rng(7)

# 2. Generate 5 random integers from 1 to 100 (inclusive)
random_integers = rng.integers(1, 101, size=5)
print("Random integers:", random_integers)

# 3. Generate a 2 × 3 array of random floats between 0 and 1
random_floats = rng.random((2, 3))
print("Random float array:\n", random_floats)