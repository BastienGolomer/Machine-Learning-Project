import numpy as np
from loss_functions import *
from grad_functions import *

## REGRESSION FUNCTIONS
    
def least_squares(y, tx, loss_function = mse):
    ''' Least squares regression, using normal equations '''

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = loss_function(y, tx, w)
    return w, loss
  
def least_square_GD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_gradient):
    ''' Least squares regression, using gradient descent '''
    w = initial_w
    for iteration in range(max_iters):

        # compute gradient
        grad = gradient(y, tx, w)  

        # compute and update w :
        w = w - gamma * grad
      
    loss = loss_function(y, tx, w)
    return w, loss
  
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_stoch_gradient): 
    ''' Least squares regression using stochastic gradient descent '''
    N = len(tx) # max number of x_n
    w = initial_w
    for iteration in range(max_iters):
        # randomly select a datapoint
        n = np.random.randint(0, N-1)  
        xn = tx[n,:]
        yn = y[n]
        # compute gradient
        grad = gradient(yn, xn, w)
        # compute and update y
        w = w - gamma * grad
        
    loss = loss_function(y, xn, w)       
    return w, loss

def ridge_regression(y, tx, lambda_, loss_function = mse):
    ''' Ridge regression using normal equations '''

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, loss_function(y, tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ''' Logistic regression using gradient descent or SGD '''
    return least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function = log_loss, gradient = log_gradient)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ''' Regularized logistic regression using gradient descent or SGD '''

    w = initial_w
    for iteration in range(max_iters):
        # Compute gradient 
        grad = log_gradient(y, tx, w) + 2 * lambda_ * w
        
        # Compute and update w :
        w = w - gamma * grad 
    m = len(y)
    loss = log_loss(y, tx, w) - lambda_/(2 * m) * sum(w)
    return w, loss
                             
