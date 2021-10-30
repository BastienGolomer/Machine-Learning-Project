import numpy as np

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
                
    return np.concatenate(y, axis=1)

## Expand the features taking sine and cosine of values, then doing polynomial expansion :
def expand_features_angles(X):
    new_x = X.copy()
    new_x = np.concatenate((np.ones((X.shape[0], 1)), X, np.sin(X), np.cos(X), np.sin(X) ** 2, np.cos(X) ** 2), axis = 1)
    return new_x       