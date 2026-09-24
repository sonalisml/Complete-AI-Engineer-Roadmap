import numpy as np
#1d array
a = np.array([1,2,3])
print(a)
print(a.ndim)
print(a.shape)
print(a.size)

b = np.array([[4,5,6], 
               [6,7,8]])
print(b)
print(b.ndim)
print(b.shape)
print(b.size)

c = np.zeros((2,3))
d = np.ones((4,4))
e = np.arange(3,10,2)
f = np.linspace(2,15,4)
print(c)
print(d)
print(e)
print(f)