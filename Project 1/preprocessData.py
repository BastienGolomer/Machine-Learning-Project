import numpy as np
from loading import *

def standardize_clean_dataframe(X):
    '''
    Input = 2 dimensional array of features
    Output = updated 2 dimensional array of features
    This function does several things 
    1) It allows to remove columns of the dataframe which have standard deviation 0 : 
    - A column with only -999 values will be deleted. 
    2) It replaces the -999 values in a column with non zero standard deviation by the median value of the column (computed without taking the -999 values into account)
    3) It standardises the data, i.e for each column we substract by the mean and divide by the standard deviation '''
    
    dataframe = X.copy()
    n = dataframe.shape[1]
    todelete = []
    for i in range(n):
        column = dataframe[:, i]
        Median = np.median(column)
        stand_dev = np.std(column)

        if stand_dev == 0:
            todelete.append(i)
            continue
        else:
            mask = np.where(dataframe[:, i] == -999)
            for j in mask:
                dataframe[j, i] = Median
    dataframe = np.delete(dataframe, todelete, axis=1)
    dataframe[:, 1:] = (dataframe[:, 1:] - np.mean(dataframe[:, 1:],
                        axis=0))/np.std(dataframe[:, 1:], axis=0)

    return dataframe
