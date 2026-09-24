import numpy as np

a = np.array([10, 20, 30, 40, 50])

# 1. Change the second element to 100
a[1] = 100
# 2. Change values greater than 35 to -1
a[a>35] = -1
# 3. Add 10 to values less than 30
a[a<30] +=10

print(a)