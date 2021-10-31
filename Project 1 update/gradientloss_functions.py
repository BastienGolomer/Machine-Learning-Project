import numpy as np

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
    return np.dot(tx.T, sigmoid(np.dot(tx, w))-y)


## THE LOGISTIC FUNCTION
# Returns the value of the logistic function, 1/(1 + exp(-x))
def sigmoid(x):
    return np.exp(x)/(1.0 + np.exp(x))

## THE LOSS FUNCTIONS

# Mean-squared error
def mse(y, tx, w):
    e = y - tx.dot(w) # compute the error vector
    return np.mean(e**2.0)/2.0

# Mean-absolute error
def mae(y, tx, w):
    e = y - tx.dot(w) # compute the error vector
    return np.mean(np.abs(e))

# Root mean-squared error
def rmse(y, tx, w):
    return np.sqrt(mse(y, tx, w))

# Logistic loss
def log_loss(y, tx, w):
    return np.sum(np.log(1.0 + np.exp(tx.dot(w))) - y * tx.dot(w))