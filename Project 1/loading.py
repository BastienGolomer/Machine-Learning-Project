import numpy as np

# Creates a function that loads the data, and outputs y (the class labels), tx (the features), and the ids
def load_csv_data(path_data):
    y = np.genfromtxt(path_data, delimiter = ",", skip_header=1, dtype = str, usecols = 1)
    x = np.genfromtxt(path_data, delimiter = ",", skip_header=1)
    ids = np.genfromtxt(path_data, delimiter = ",", skip_footer=(250000), dtype= str)
    
    input_data = x[:, 2:]
    
    # Y has class labels, we convert them to 1 or -1 to be able to manipulate them in an easier way.
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0
    
    return yb, input_data, ids

def write_csv(ids, predictions, name_csv):
### Creates an output submission in a csv file to AICrowd
'''ids : id of the event of the associated prediction
   predictions : class label of the variable y 
   name_csv : name of the csv file in which we write our results '''
    with open(name, newline = ' ') as csvfile:
        names = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter = ' ', fieldnames = names)
        writer.writeheader()
        for x, y in ids, prediction:
            writer.writerow({"Id" : int(x), "Prediction" : int(predictions)})
