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
    b = sigmoid(tx.dot(w)) - y
    return tx.T.dot(b)
  
# Sigmoid gradient
def calc_gradient_sigm(y, tx, w):
    e = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(e)/len(e)
    return grad

# Stochastic gradient
def calc_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = - tx.T.dot(e)
    return grad