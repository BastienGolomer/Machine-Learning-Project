import numpy as np
from loss_functions import *
from grad_functions import *

## REGRESSION FUNCTIONS
    
# Least squares regression, using normal equations :
def least_squares(y, tx, loss_function = mse):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = loss_function(y, tx, w)
    return w, loss
  
# Least squares regression, using gradient descent :
def least_square_GD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_gradient):
    w = initial_w
    for iteration in range(max_iters):
        
        # compute gradient
        grad = gradient(y, tx, w)  

        # compute and update w :
        #print(w)
        #print(grad)
        w = w - gamma * grad
      
    loss = loss_function(y, tx, w)
    return w, loss
  
# Least squares regression using stochastic gradient descent :
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_stoch_gradient): 
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

# Ridge regression using normal equations :
def ridge_regression(y, tx, lambda_, loss_function = mse):
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, loss_function(y, tx, w)

# Logistic regression using gradient descent or SGD :
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function = log_loss, gradient = log_gradient)

# Regularized logistic regression using gradient descent or SGD :
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for iteration in range(max_iters):
        # Compute gradient 
        grad = log_gradient(y, tx, w) + 2 * lambda_ * w
        
        # Compute and update w :
        w = w - gamma * grad 
    m = len(y)
    loss = log_loss(y, tx, w) - lambda_/(2 * m) * sum(w)
    return w, loss
                             
