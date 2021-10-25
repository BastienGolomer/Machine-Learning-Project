import numpy as np

## SIGMOID
def sigmoid(x):
    return 1/(1 + np.exp(-x))

## THE LOSS FUNCTIONS

# Mean-squared error
def mse(y, tx, w):
  e = y - tx.dot(w) # compute the error vector
  return np.mean(e**2)/2

# Mean-absolute error
def mae(y, tx, w):
  e = y - tx.dot(w) # compute the error vector
  return np.mean(np.abs(e))

# Root mean-squared error
def rmse(y, tx, w):
  return np.sqrt(mse(y, tx, w))

# Logistic loss
def log_loss(y, tx, w):
    return np.sum(y * np.log(sigmoid(tx.dot(w))) + (1 - y) * np.log(1 - sigmoid(tx.dot(w))))/len(y)

