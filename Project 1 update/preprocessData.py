import numpy as np
from proj1_helpers import *

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
    for i in range(n):
        column = dataframe[:, i].copy()      
        column = np.where(column == -999, np.median(column[column != -999]), column)
        dataframe[:, i] = column.copy()
        
    Mean = np.mean(dataframe, axis = 0)
    stand_dev = np.std(dataframe, axis = 0)
    
    matrix_of_mean = np.full(dataframe.shape, Mean)
    matrix_of_std = np.full(dataframe.shape, stand_dev)
    return (dataframe - matrix_of_mean)/(matrix_of_std)