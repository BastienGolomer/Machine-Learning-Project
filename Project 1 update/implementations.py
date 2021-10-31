import numpy as np
from gradientloss_functions import *

## REGRESSION FUNCTIONS
    
def least_squares(y, tx, loss_function = mse):
    ''' Least squares regression, using normal equations '''
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    loss = loss_function(y, tx, w)
    return w, loss
  
def least_squares_GD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_gradient):
    ''' Least squares regression, using gradient descent '''
    w = initial_w
    for iteration in range(max_iters):
        # compute gradient
        grad = gradient(y, tx, w)  
        # compute and update w :
        w = w - gamma * grad
      
    loss = loss_function(y, tx, w)
    return w, loss

  
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function = mse, gradient = calc_gradient): 
    ''' Least squares regression using stochastic gradient descent '''
    N = tx.shape[1] # max number of x_n
    w_t = initial_w
    losses = []
    
    for iteration in range(max_iters):
        # randomly select a datapoint
        n = np.random.randint(0, N)  
        xn = tx[n]
        yn = y[n]
        # compute gradient
        grad = gradient(yn, xn, w_t)
        # compute and update y
        w_t = w_t - gamma * grad
        losses.append(loss_function(yn, xn, w_t))
    
    loss = 1.0/N*sum(losses)
   
    return w_t, loss

def ridge_regression(y, tx, lambda_, loss_function = mse):
    ''' Ridge regression using normal equations '''

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, loss_function(y, tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for iteration in range(max_iters):
        # Compute gradient and loss
        grad = log_gradient(y, tx, w)
        loss = log_loss(y, tx, w)
        
        # Compute and update w
        w = w - gamma * grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ''' Regularized logistic regression using gradient descent or SGD '''
    w = initial_w
    loss = 0
    m = len(y)
    for iteration in range(max_iters):
        # Compute gradient  and loss
        grad = log_gradient(y, tx, w) + lambda_ * np.linalg.norm(w)
        loss = log_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
        # Compute and update w :
        w = w - gamma * grad 
    return w, loss/(2 * m)

# Function that predicts values given a data matrix tx and weights w :
def predict_values(tx, w):
    y = np.dot(tx, w)
    y[np.where(y > 0)] = 1
    y[np.where(y <= 0)] = -1
    return y  

# Compute the confusion matrix between two vectors, used to compare models.
def compute_confusion_matrix(true_values, predicted_values):
    '''Computes a confusion matrix using numpy for two np.arrays
    true_values and predicted_values. '''
    K = len(np.unique(true_values)) # Number of classes 
    result = np.zeros((K, K))
    true_values[np.where(true_values == -1)] = 0
    predicted_values[np.where(predicted_values == -1)] = 0

    for i in range(len(true_values)):
        result[int(true_values[i])][int(predicted_values[i])] += 1
        
    return result

def batch_iter(y, tx, batch_size, num_batches = 1, shuffle = True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
        Function provided in the helpers of lab2
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio = 0.8, seed = 1):
    """split the dataset based on the split ratio - taken from lab4"""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row)) 
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te