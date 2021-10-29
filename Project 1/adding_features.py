import numpy as np

## We consider that the relationship between the features is more complex than a simple linear relationship. To deal with this complexity
## and to avoid underfitting, we add cross-terms and polynomial expanding.

## expands the array X to the given degree
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree - 
        using the method implemented in lab 2 """
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


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
