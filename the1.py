import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

# Random seed
np.random.seed(499)

# Create np array from csv
data = np.genfromtxt(fname='pima-indians-diabetes.csv', delimiter=',', dtype=float)

def train_test_split(X_Y, train_ratio=0.7, first_n_element=0, shuffle=False):
    X_Y_copy = np.copy(X_Y)
    if shuffle:
        np.random.shuffle(X_Y_copy)
    if first_n_element == 0:
        first_n_element = X_Y_copy.shape[0] * train_ratio
    X_train, X_test, Y_train, Y_test = X_Y_copy[:first_n_element, :-1], X_Y_copy[first_n_element:, :-1], X_Y_copy[:first_n_element, -1], X_Y_copy[first_n_element:, -1] 
    return X_train, X_test, Y_train, Y_test
def sigmoid(w_t_x):
    h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    return h_w_x
def hyptothesis(x, w):
    w_t_x = 0.0
    exp_x = expanded_x(x)
    for i in range(len(exp_x)):
        w_t_x += exp_x[i] * w[i]
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
def logistic_regression(X, Y, alpha, epochs=10):
    m = len(Y)
    w = np.random.randn(len(X[0]) + 1)
    for i in range(epochs):
        updated_w = gradient_descent(X, Y, w, alpha)
        w = updated_w
        if i % 10 == 0:
            print 'Current w: ', w
            print 'Current cost: ', cost_function(X, Y, w)
