import numpy as np

from implementations import *
from preprocessData import *
from gradientloss_functions import *
from proj1_helpers import *
from features import *

data_train_path = './train.csv'
data_test_path = './test.csv'
data_output_path = './output.csv'

# Loading data
print('Loading data...')
y, tx, ids = load_csv_data(data_train_path)
y_test, tx_test, ids_test = load_csv_data(data_test_path)

# Preprocessing data
print('Processing data...')
tx = standardize_clean_dataframe(tx)
tx_test = standardize_clean_dataframe(tx_test)

# Augmenting features :
print('Learning...')
tx = expand_features_angles(tx)
tx_test = expand_features_angles(tx_test)

# Splitting the data between training and validation
print('Splitting dataset between training and validation subsets...')
x_tr, x_val, y_tr, y_val = split_data(tx, y)

# Computing weights using regularized logistic regression :
print('Calculating result')
initial_w = np.zeros(tx.shape[1])
w, _ = logistic_regression(np.where(y_tr == -1, 0, y_tr), x_tr, initial_w, 125, 0.01)


# Writing labels
print('Presenting results')
y_pred = predict_labels(w, tx_test)
print(y_pred)

# Outputting the result in a csv file for AIcrowd submission
print('Writing the output')
create_csv_submission(ids_test, y_pred, data_output_path)

# Showing the confusion matrix for the validation set :
print('The confusion matrix of the validation set is :')
y_pred_test = predict_labels(w, x_val)
print(compute_confusion_matrix(y_val, y_pred_test))

# Computing the loss
print('The loss is :')
loss = mse(y_val, x_val, w)
print(loss)
