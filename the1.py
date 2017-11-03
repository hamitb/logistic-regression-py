import numpy as np
import matplotlib.pyplot as plt
from math import exp

### Create np array from csv
data = np.genfromtxt(fname='pima-indians-diabetes.csv', delimiter=',', dtype=float)

def sigmoid(x, w):
    expanded_x = np.insert(x, 0, 1.0)
    w_t_x = 0.0

    for i in range(len(x)):
        w_t_x += expanded_x[i] * w[i]

    h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    return h_w_x

def estimate_y_hat(h_w_x):
    if h_w_x >= 0.5:
        return 1.0
    else:
        return 0.0

