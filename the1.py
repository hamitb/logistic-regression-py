import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

# Random seed
np.random.seed(499)

# Create np array from csv
data = np.genfromtxt(fname='pima-indians-diabetes.csv', delimiter=',', dtype=float)

def sigmoid(w_t_x):
    h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    return h_w_x

def hyptothesis(x, w):
    w_t_x = 0.0
    for i in range(len(x)):
        w_t_x += x[i] * w[i]
    return w_t_x

def expanded_x(x):
    exp_x = np.insert(x, 0, 1)
    return exp_x

def estimate_y_hat(h_w_x):
    if h_w_x >= 0.5:
        return 1.0
    else:
        return 0.0

def cost_function(X, Y, w):
    sum_of_errors = 0.0
    m = len(X)
    for i in range(m):
        y_i = Y[i]
        x_i = X[i]
        h_w_x = hyptothesis(x_i, w)
        sum += y_i * log(h_w_x) + (1-y_i) * log(1 - h_w_x)

    j_w = (-1.0 / m) * sum_of_errors
    return j_w

def cost_function_derivative(X, Y, w, j):
    sum_of_errors = 0
    m = len(X)
    for i in range(m):
        x_i = X[i]
        x_i_j = x_i[j]
        y_i = Y[i]
        h_w_x = hyptothesis(x_i, w)
        sum_of_errors += (h_w_x - y_i) * x_i_j
    derivative_of_j = (1.0 / m) * sum_of_errors
    return derivative_of_j
    
def gradient_descent(X, Y, w, alpha):
    updated_w = np.array([])
    for i in range(len(w)):
        cf_derivative = cost_function_derivative(X, Y, w, i)
        updated_w_i = w[i] - alpha * cf_derivative
        updated_w.append(updated_w_i)
    return updated_w