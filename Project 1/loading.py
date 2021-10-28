import numpy as np
import csv

# Creates a function that loads the data, and outputs y (the class labels), tx (the features), and the ids
def load_csv_data(path_data):
    y = np.genfromtxt(path_data, delimiter = ",", skip_header = 1, dtype = str, usecols = 1)
    x = np.genfromtxt(path_data, delimiter = ",", skip_header = 1)
    labels = np.genfromtxt(path_data, delimiter = ",", skip_footer = (250000), dtype = str)
    
    input_data = np.delete(x, 1, axis = 1)
    
    # y has class labels, we convert them to 1 or 0 to be able to manipulate them in an easier way.
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0
    
    return yb, input_data, labels

def write_csv(ids, predictions, name_csv):
### Creates an output submission in a csv file to AICrowd
    '''ids : id of the event of the associated prediction
    predictions : class label of the variable y 
    name_csv : name of the csv file in which we write our results '''
    with open(name_csv, 'w', newline = '') as csvfile:
        names = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter = ',', fieldnames = names)
        writer.writeheader()
        for x, y in zip(ids, predictions):
            writer.writerow({"Id" : int(x), "Prediction" : int(y)})



def update_dataframe_median (X) : 
    dataframe = X.copy()
    n = dataframe.shape[1]-1
    todelete = []
    for i in range(n) :
        column = dataframe[:,i]
        Median = np.median(column)

        if Median == -999 :
            todelete.append(i)
            continue 
        else :
            mask = np.where(dataframe[:,i] == -999 )
            for j in mask :
                dataframe[j,i] = Median

    dataframe = np.delete(dataframe, todelete, axis = 1)

    return dataframe

        
