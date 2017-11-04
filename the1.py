import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

# Random seed
np.random.seed(499)

def normalize(raw_data):
    normalized_data = np.copy(raw_data)
    num_of_rows = raw_data.shape[0]
    num_of_columns = raw_data.shape[1]
    for j in range(num_of_columns-1):
        column_min, column_max = min(raw_data[:, j]), max(raw_data[:, j])
        column_range = column_max - column_min
        for i in range(num_of_rows):
            normalized_data[i][j] = (normalized_data[i][j] - column_min) / column_range
    return normalized_data
def train_test_split(X_Y, train_ratio=0.7, first_n_element=0, shuffle=False):
    X_Y_copy = np.copy(X_Y)
    if shuffle:
        np.random.shuffle(X_Y_copy)
    if first_n_element == 0:
        first_n_element = X_Y_copy.shape[0] * train_ratio
    X_train, X_test, Y_train, Y_test = X_Y_copy[:first_n_element, :-1], X_Y_copy[first_n_element:, :-1], X_Y_copy[:first_n_element, -1], X_Y_copy[first_n_element:, -1] 
    return X_train, X_test, Y_train, Y_test
def sigmoid(x, w):
    w_t_x = hyptothesis(x, w)
    try:
        h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    except OverflowError:
        h_w_x = 0.0
    return h_w_x
def expanded_x(x):
    return np.insert(x, 0, values=1, axis=1)
def hyptothesis(x, w):
    w_t_x = 0.0
    for i in range(len(x)):
        w_t_x += x[i] * w[i]
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
        h_w_x = sigmoid(x_i, w)
        sum_of_errors += y_i * log(h_w_x) + (1 - y_i) * log(1 - h_w_x)

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
        updated_w = np.append(updated_w, updated_w_i)
    return updated_w
def logistic_regression(X, Y, alpha, X_test, Y_test, epochs=10):
    m = len(Y)
    X = expanded_x(X)
    w = np.random.randn(len(X[0]))
    for i in range(epochs):
        updated_w = gradient_descent(X, Y, w, alpha)
        w = updated_w
        Y_train_pred = predict(X,w)
        Y_test_pred = predict(X_test, w)
        if i % 10 == 0:
            # print 'Current w: ', w
            print 'Train accuracy:', accuracy(Y_train_pred, Y), ',Test accuracy:', accuracy(Y_test_pred, Y_test)
def predict(X, w):
    Y_pred = np.zeros(X.shape[0])
    for i in range(len(X)):
        h_w_x = sigmoid(X[i], w)
        if h_w_x >= 0.5:
            Y_pred[i] = 1.0
        else:
            Y_pred[i] = 0.0
    return Y_pred
def accuracy(Y_pred, Y):
    loss_01 = 0.0
    for i in range(len(Y_pred)):
        if Y_pred[i] != Y[i]:
            loss_01 += 1
    acc = (1.0 - loss_01 / len(Y_pred)) * 100
    return acc

if __name__ == '__main__':
    data = np.genfromtxt(fname='pima-indians-diabetes.csv', delimiter=',', dtype=float)
    data = normalize(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data, first_n_element=668)
    
    logistic_regression(X_train, Y_train, 1e-2, epochs=1000, X_test=X_test, Y_test=Y_test)
    
