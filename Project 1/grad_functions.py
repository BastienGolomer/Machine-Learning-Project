import numpy as np
from loss_functions import *

## GRADIENT FUNCTIONS

# Classical gradient
def calc_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/len(e)
    return grad
  
# Logistic gradient
def log_gradient(y, tx, w):
    w[0] = np.sum(sigmoid(tx.dot(w)) - y) * tx[0]
    w[1:end] = np.sum(sigmoid(tx.dot(w)) - y) * tx[1:end] + lambda_/len(w[1:end]) * w
    return w
  
# Sigmoid gradient
def calc_gradient_sigm(y, tx, w):
    e = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(e)/len(e)
    return grad
  
  
