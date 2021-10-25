import numpy as np

# Creates a function that loads the data, and outputs y (the class labels), tx (the features), and the ids
def load_csv_data(path_data):
    y = np.genfromtxt(path_data, delimiter = ",", skip_header = 1, dtype = str, usecols = 1)
    x = np.genfromtxt(path_data, delimiter = ",", skip_header = 1)
    ids = x[:, 0].astype(np.int)
    
    input_data = x[:, 2:]
    
    # Convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    
    return yb, input_data, ids
