import numpy as np

#==========
# Activation function
#==========
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1/(1+np.exp(-z))


#==========
# Input Layer declaration
#==========
x = np.array([35,6000,1,8])
#==========
# Weight and bias using np.random.rand function
#==========
W1 = np.random.rand(4,5)
b1 = np.random.rand(5)

print("\nWeight Matrix (Input → Hidden)")
print(W1)

print("\nBias")
print(b1)

z1 = np.dot(x, W1) + b1
print("z1", z1)
A1 = relu(z1)
print("a1", A1)

##=====================##
w2 = np.random.rand(5)
b2 = np.random.rand(1)

z2 = np.dot(A1, w2) + b2

predictions  = sigmoid(z2)

if predictions >0.5:
    print("this is ")
else:
    print("this is not")