import numpy as np

## We consider that the relationship between the features is more complex than a simple linear relationship. To deal with this complexity
## and to avoid underfitting, we add cross-terms and polynomial expanding.

## expands the array X to the given degree
def expand_polynomial(X, degree):
    
    y = [X.copy()]
    for d in range(2, degree + 1):
        y.append(np.power(X, d))
    return np.concatenate(y)


## Augments the features by adding cross-terms and polynomial expand
def add_features(X, degree):
    new_x = X.copy()
    
    # adding the polynomial terms
    new_x = expand_polynomial(X, degree)
    
    # adding the cross terms
    top = X.shape[1]
    y = [new_x]
    for i in range(top):
        for j in range(i+1, top):
            y.append((X[:,i] * X[:,j]).reshape(-1, 1)) 
            
    
    return np.concatenate(to_concatenate, axis=1)
											 
