import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp,log
from the1 import *
sns.set(style='ticks', palette='Set2')

# Set graph limits
plt.xlim((-10, 10))

# Create data array
X = np.array(range(101), dtype=float)
X = np.reshape(X, (len(X), 1))
X /= 5.0
X -= 10.0

# Expand X and add bias
X = np.insert(X, 0, values=1, axis=1)

# Create a random weight vector
w1 = np.array([-5, 1])
w2 = np.array([0, 1])
w3 = np.array([5, 1])

# Draw the plots of input vs sigmoid while changing bias
plt.plot(X[:, 1], map(lambda x: sigmoid(x, w1), X))
plt.plot(X[:, 1], map(lambda x: sigmoid(x, w2), X))
plt.plot(X[:, 1], map(lambda x: sigmoid(x, w3), X))
plt.ylabel("Sigmoid function's output")
plt.xlabel("Input feature")

plt.legend(['bias = -5', 'bias = 0', 'bias = +5'])
plt.savefig('bias.png')
plt.gcf().clear()
