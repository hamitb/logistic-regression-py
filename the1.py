import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

### Create np array from csv
data = np.genfromtxt(fname='pima-indians-diabetes.csv', delimiter=',', dtype=float)

def sigmoid(w_t_x):
    h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    return h_w_x

def hyptothesis(x, w):
    expanded_x = np.insert(x, 0, 1.0)
    w_t_x = 0.0
    for i in range(len(x)):
        w_t_x += expanded_x[i] * w[i]
    return w_t_x

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

