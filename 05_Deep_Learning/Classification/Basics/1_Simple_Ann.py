#Program 1- Simple ANN program
import numpy as np

inputs = np.array([2,1,5])

weights = np.array([0.2,0.1,0.3])

bias = 0.5

z = np.dot(inputs,weights)+bias

def relu(z):
    return np.maximum(0,z)

def sigmoid(z):
    return 1/(1+ np.exp(-z))


print("Relu output is ", relu(z))
print("Sigmoid out put is ", sigmoid(z))
#================================
#Program 2- Simple House prediction ANN program
#================================
print("\n")
print("="*60)
print("Program-2")
print("="*60)
h = np.array([2000,3,2])# area , bedroom, bathroom
w = np.array([0.7,0.5,0.2]) # weights accordingly
b = 0.5  #bias
z1 = np.dot(h,w)+b # calculating z via dot product

print("After Relu activation", relu(z1))
print("After sigmoid activation",  sigmoid(z1))
