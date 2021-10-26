import numpy as np
from loss_functions import *

## GRADIENT FUNCTIONS

# Classical gradient
def calc_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/len(e)
    return grad
  
# Logistic gradient
def log_gradient(y, tx, w, lambda_ = 0):
    w[0] = np.sum(sigmoid(tx.dot(w)) - y)*(tx[0])
    w[1:] = np.sum(sigmoid(tx[1:].dot(w[1:])) - y) * tx[1:] + lambda_/len(w[1:]) * w[1:]
    return w
  
# Sigmoid gradient
def calc_gradient_sigm(y, tx, w):
    e = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(e)/len(e)
    return grad
  