import numpy as np
import csv

def load_csv_data(path_data):
    ''' 
    Input = pathdata : path to the csv file to be read
    Output = 
    - yb = response variable
    - input_data : array of features and the Ids of each feature
    - labels = the labels of each column of the csv file.

    This function reads and separates the interesting data from a .csv file.
    '''
    y = np.genfromtxt(path_data, delimiter=",",
                      skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(path_data, delimiter=",", skip_header=1)
    labels = np.genfromtxt(path_data, delimiter=",",
                           skip_footer=(250000), dtype=str)

    input_data = np.delete(x, 1, axis=1)

    # y has class labels, we convert them to 1 or 0 to be able to manipulate them in an easier way.
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0

    return yb, input_data, labels


def write_csv(ids, predictions, name_csv):
    '''
    Inputs = 
    - ids : id of the event of the associated prediction
    - predictions : class label of the variable y 
    - name_csv : name of the csv file in which we write our results 
    
    Outputs a .csv file acceptable by AICrowd '''
    with open(name_csv, 'w', newline='') as csvfile:
        names = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=names)
        writer.writeheader()
        for x, y in zip(ids, predictions):
            writer.writerow({"Id": int(x), "Prediction": int(y)})


def update_dataframe_median(X):
    '''
    Input = 2 dimensional array of features
    Output = updated 2 dimensional array of features

    This function does several things 
    1) It allows to remove columns of the dataframe which have standard deviation = 0 : 
    - A column with only -999 values will be deleted. 
    - A column with always the same value (which is the case when the dataframe is divided following PRI_jet_num) would also be deleted
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