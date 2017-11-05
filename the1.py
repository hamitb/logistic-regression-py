import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp,log
sns.set(style='ticks', palette='Set2')

# Random seed
np.random.seed(499)

def normalize(raw_data, algorithm="min-max"):
    normalized_data = np.copy(raw_data)
    if algorithm == "min-max":
        num_of_rows = raw_data.shape[0]
        num_of_columns = raw_data.shape[1]
        for j in range(num_of_columns-1):
            column_min, column_max = min(raw_data[:, j]), max(raw_data[:, j])
            column_range = column_max - column_min
            for i in range(num_of_rows):
                normalized_data[i][j] = (normalized_data[i][j] - column_min) / column_range
    elif algorithm == "mean-std":
        normalized_data = (raw_data[:, :-1] - np.mean(raw_data[:, :-1], axis=0)) / np.std(raw_data[:, :-1], axis=0)
        normalized_data = np.insert(normalized_data, normalized_data.shape[1], values=raw_data[:, -1], axis=1)
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
    w_t_x = 0.0
    for i in range(len(x)):
        w_t_x += x[i] * w[i]
    h_w_x = 1.0 / (1.0 + exp(-w_t_x))
    return h_w_x
def expanded(x):
    return np.insert(x, 0, values=1)
def cost_function(X, Y, w):
    sum_of_errors = 0.0
    m = len(X)
    for i in range(m):
        h_w_x = sigmoid(expanded(X[i]), w)
        sum_of_errors += Y[i] * log(h_w_x) + (1 - Y[i]) * log(1 - h_w_x)
    j_w = (-1.0 / m) * sum_of_errors
    return j_w
def cost_function_derivative(X, Y, w):
    derivative_of_j = np.zeros(w.shape)
    m = len(X)
    for i in range(m):
        h_w_x = sigmoid(expanded(X[i]), w)
        derivative_of_j += (h_w_x - Y[i]) * expanded(X[i])
    derivative_of_j *= (1.0 / m)
    return derivative_of_j
def gradient_descent(w, alpha, derivative_of_j):
    return w - alpha * derivative_of_j
def logistic_regression(X, Y, alpha, X_test, Y_test, epochs=10):
    info = np.empty((0,4), dtype=float)
    w = np.random.randn(len(X[0]) + 1)
    for i in range(epochs):
        derivative_of_j = cost_function_derivative(X, Y, w)
        w = gradient_descent(w, alpha, derivative_of_j)
        Y_train_pred = predict(X, w)
        Y_test_pred = predict(X_test, w)
        info = np.append(info, np.array([[i, accuracy(Y_train_pred, Y),\
                            accuracy(Y_test_pred, Y_test), cost_function(X, Y, w)]]), axis=0)
        if i % 10 == 0:
            print 'Epoch:', i, '/', epochs, ",", 'Train accuracy:', accuracy(Y_train_pred, Y),\
                'Test accuracy:', accuracy(Y_test_pred, Y_test), "Cost:", cost_function(X, Y, w)
    return info
def predict(X, w):
    Y_pred = np.zeros(X.shape[0])
    for i in range(len(X)):
        h_w_x = sigmoid(expanded(X[i]), w)
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
def read_csv_to_np(fname, normalize_data=False, normalize_algo='mean-std'):
    data = np.genfromtxt(fname=fname, delimiter=',', dtype=float)
    if normalize_data:
        return normalize(data, algorithm=normalize_algo)
    else:
        return data

if __name__ == '__main__':
    data = read_csv_to_np(fname="pima-indians-diabetes.csv", normalize_data=True)
    X_train, X_test, Y_train, Y_test = train_test_split(data, first_n_element=668)
    # Apply logistic regression and get information about process
    info = logistic_regression(X_train, Y_train, 0.1, epochs=10000, X_test=X_test, Y_test=Y_test)
    # Drawing cost function vs epoch graph
    plt.plot(info[:, 0], info[:, 3])
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    sns.despine()
    plt.savefig('cost-epoch.png')
    plt.gcf().clear()
    # Drawing training set accuracy vs epoch graph
    plt.plot(info[:, 0], info[:, 1])
    plt.ylabel("Training set accuracy")
    plt.xlabel("Epoch")
    sns.despine()
    plt.savefig('train_accuracy-epoch.png')
    plt.gcf().clear()
    # Drawing test set accuracy vs epoch graph
    plt.plot(info[:, 0], info[:, 2])
    plt.ylabel("Test set accuracy")
    plt.xlabel("Epoch")
    sns.despine()
    plt.savefig('test_accuracy-epoch.png')
    plt.gcf().clear()