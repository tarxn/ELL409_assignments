import random
import numpy as np
import get_data
# print(get_data.X_subset.shape)
# print(get_data.y_subset.size)
# print(get_data.y_subset[0:10])

# def sigmoid(x,w,b):       # takes single datapoint x
#     z=x*w.transpose() +b
#     s = 1/(1+np.exp(-z))
#     return
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(w, x):
    # Computes the weighted sum of inputs
    # print(x.size,w.size)
    return np.dot(x, w)

def probability(w, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(w, x))

def cost_function(w, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    y_alt= (y%9)/2
    total_cost = -(1 / m) * np.sum(y_alt * np.log(probability(w, x)) + (1 - y_alt) * np.log(
                    1 - probability(w, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.transpose(), sigmoid(net_input(theta,   x)) - y)

def prediction(w,x):
    p=probability(w,x)
    for i in range(p.size):
        if(p[i]>=0.5):
            p[i]=1
        else:
            p[i]=0
    return p

def error(w,x,y):
    p=prediction(w,x)
    m=y.size
    y=(y%9)/2
    error=0
    for i in range(p.size):
        if(p[i]!=y[i]):
            error=error+1
    error=error/m
    return error
def stochastic_gd(sample, label, num_it, learn_rate):
    n=label.shape[0]
    m=sample.shape[1]
    label=(label%9)/2
    w=np.ones(m)
    for i in range(num_it):
        nth=random.randint(0,int(n-1))
        xth=sample[nth:nth+1, :]
        yth= label[nth:nth+1]
        w=w-learn_rate*gradient(w,xth,yth)
    return w

