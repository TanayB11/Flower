#My first neural network!
#from matplotlib import pyplot as plt
import numpy as np
import os

# Network Structure
#     o     flower type
#    / \    w1, w2, b
#   o   o   length, width


#Input
mystery_flower = [4.5, 1]

#Data in format of length, width, type (0 for blue/1 for red)
data = [[3,  1.5, 1],
        [2,  1,   0],
        [4,  1.5, 1],
        [3,    1, 0],
        [3.5, .5, 1],
        [2,   .5, 0],
        [5.5,  1, 1],
        [1,    1, 0]]

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Sigmoid prime
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))


#T = np.linspace(-6, 6, 100)
#Y = sigmoid(T)
#plt.plot(T, sigmoid(T), c='r')
#plt.plot(T, sigmoid_p(T), c='b')


learning_rate = 0.2
costs = []


#Random numbers for the weights and bias
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

#Training loop
for i in range(1000):
    ri = np.random.randint(len(data))
    point = data[ri]
    
    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    #Defining the target and cost/loss variables
    target = point[2]
    cost = np.square(pred - target)
        
    #Derivative of the cost in respect to the prediction
    dcost_pred = 2 * (pred - target)
    dpred_dz = sigmoid_p(z)

    #Derivatives of the ... in respect to ...
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_dz
    
    #Chain rule
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    #Adjust weights and bias
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

    if i % 100 == 0:
        cost_sum = 0
        for j in range(len(data)):
            point = data[j]
            z = point[0] * w1 +point[1] * w2 + b
            pred = sigmoid(z)           
        costs.append(cost_sum/len(data))
        
z = mystery_flower[0] * w1 +mystery_flower[1] * w2 + b
pred = sigmoid(z)

print("pred: {}".format(pred))

def which_flower(length,width):
    z = length * w1 + length * w2 + b
    pred = sigmoid(z)
    if pred < .5:
        os.system("say blue")
    else:
        os.system("say red")

which_flower(4, 5)
    
    
