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
print(tx.shape)
print('Learning...')
# tx = add_features(tx, 2)
tx_copy=tx.copy()
tx = expand_features_angles(tx)
newcol,indexs=add_dim(y,tx_copy,3)
tx=np.concatenate((tx,newcol),axis=1)
# tx_test = add_features(tx_test, 2)
tx_test = expand_features_angles(tx_test)
for i in indexs:
    tx_test=add_col(tx_test,i[0],i[1])


# Splitting the data between training and validation
print('Splitting dataset between training and validation subsets...')
x_tr, x_val, y_tr, y_val = split_data(tx, y, 0.9)

# Computing weights using regularized logistic regression :
print('Calculating result')
initial_weights = np.zeros(tx.shape[1])
w, _ = reg_logistic_regression(np.where(y_tr == -1, 0, y_tr), x_tr, 0, initial_weights, 100, 1e-6)

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
