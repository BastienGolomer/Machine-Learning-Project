import numpy as np
from loss_functions import *

## GRADIENT FUNCTIONS

# Classical gradient
def calc_gradient(y, tx, w):
    ''' Computes the regular gradient '''
    e = y - tx.dot(w)
    grad = - 1.0/y.shape[0] * tx.T.dot(e)
    return grad
  
# Logistic gradient
def log_gradient(y, tx, w):
    ''' Computes the gradient using a sigmoid function'''
    e = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(e)
    return grad
  
# Sigmoid gradient
def calc_gradient_sigm(y, tx, w):
    ''' Computes the gradient using a sigmoid function, normalised'''
    e = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(e)
    return grad

# Stochastic gradient
def calc_stoch_gradient(y, tx, w):
    ''' Computes the stochastic gradient'''
    e = y - tx.dot(w)
    grad = - tx.T.dot(e)
    return grad
