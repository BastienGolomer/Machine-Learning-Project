import numpy as np
from implementations import *
from loss_functions import *

## We consider that the relationship between the features is more complex than a simple linear relationship. To deal with this complexity
## and to avoid underfitting, we add cross-terms and polynomial expanding.

## expands the array X to the given degree
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree - 
        using the method implemented in lab 3 """
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return np.delete(poly, 0, axis = 1)


## Augments the features by adding cross-terms and polynomial expand
def add_features(X, degree):
    new_x = X.copy()
    
    # adding the polynomial terms
    new_x = build_poly(X, degree)
    
    # adding the cross terms
    top = X.shape[1]
    y = [new_x]
    for i in range(top):
        for j in range(i+1, top):
            y.append((X[:,i] * X[:,j]).reshape(-1, 1)) 
            
    
    return np.concatenate(to_concatenate, axis=1)


def add_col(X, col,col2):
    new_x = X.copy()
    top = X.shape[1]
    y = [new_x]
    y.append((X[:,col2] * X[:,col]).reshape(-1, 1))           
    return np.concatenate(y, axis=1)


def add_dim(y,new_X,dim):
    #will iter dim times adding a column of cross terms (can be the same column-> polynomial)
    [w, loss_] = ridge_regression(y, new_X, 0.4)
    loss_validation=loss_
    for k in range(dim):
        keepi=[0,-1,-1]
        #iters on all the combinations of columns (~30*30)
        for i in range (new_X.shape[1]-1):
            for j in range (i,new_X.shape[1]-1):
                #adds a column then train, make pred compute the loss on validate and compare this loss to the other candidates 
                Xtemp=add_col(new_X, i,j)
                X_train, X_validate = np.split(Xtemp,[int(.7*len(new_X))])
                y_train, y_validate = np.split(y,[int(.7*len(y))])
                [w, loss_] = ridge_regression(y_train, X_train,0.4)
                loss_v=(mse(y_validate,X_validate,w))
                #compare the loss
                if (keepi[0]<(loss_validation-loss_v)):
                    keepi=[loss_validation-loss_v,i,j]
        #adds the column who has the strongest loss decrease
        new_X=add_col(new_X,keepi[1],keepi[2])
        X_train, X_validate = np.split(new_X,[int(.7*len(new_X))])
        y_train, y_validate = np.split(y,[int(.7*len(y))])
        #fits a new model and compute new loss
        [w_final, loss_validation] = ridge_regression(y_train, X_train,0.4)
        losses=(mse(y_validate,X_validate,w_final))
        print (keepi[1],keepi[2],losses)
    return new_X
