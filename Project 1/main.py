import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import K_fold as KF
import adding_features as af
import implementations as imp
import split_fit as sf
from confusion_matrix import *
from loss_functions import *


# methods that can be used as regressions
methods = ['LeastSquare', 'LeastSquareGD', 'LeastSquareSGD', 'RidgeRegression', 'LogisticRegression', 'RegLogisticRegression'  ]

# load data
y,X,labels =ld.load_csv_data("./train.csv")
print('X original = ' + str(X.shape))

# isolate the identifiers of each feature
Ids = X[:,0]

# remove the Ids from X
new_X = np.delete(X,0,axis=1)
print('new_X = ' + str(new_X.shape))

# remove the lables of id and y from the "labels", to keep the relevant headers for the features in X
labels = np.delete(labels,[0,1]) 

# Now split following the PRI_jet_num, keeping the identifiers in the right order
# K = 4 # the number of subgroup desired
# w_final, loss_validation = sf.fit(y, X, Ids, K)

# ============================ TEST NO KFOLD ======================================================

new_X = ld.standardize_clean_dataframe(new_X)
print(new_X.max())

labels = np.delete(labels,[0,1]) # to keep the relevant headers for the features in X

# split the train.csv into a training set and a validation set
X_train, X_validate = np.split(new_X,[int(.9*len(new_X))])
y_train, y_validate = np.split(y,[int(.9*len(y))])

#add columns of ones
# X_train = np.column_stack((X_train, np.ones(X_train.shape[0]).T))
# X_validate = np.column_stack((X_validate, np.ones(X_validate.shape[0]).T))

n = X_train.shape[1]
percentage_max = 0
max_i = 0
for i in np.logspace(-3,5,1000) : 
    # w_temp, loss = imp.least_squares(y_train, X_train) # WORKS
    # w_temp, loss = imp.least_squares_GD(y_train,X_train,np.ones(n)/10,100,0.1) # WORKS
    # w_temp, loss = imp.least_squares_SGD(y_train,X_train,np.ones(n)/10,100,0.3) # WORKS
    # w_temp, loss = imp.ridge_regression(y_train,X_train,i) # WORKS, optimal hyperparameter = 0.010858585858585859
    # w_temp, loss = imp.logistic_regression(y_train,X_train,np.ones(n)/10,100,0.1)
    # w_temp, loss = imp.reg_logistic_regression(y_train,X_train,0.1,np.ones(n)/10,100,1.1)

    confusion_matrix = compute_confusion_matrix(y_validate, X_validate.dot(w_temp))
    percentage = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    if percentage > percentage_max :
        percentage_max = percentage
        max_i = i

print('optimal hyperparameter = ' + str(max_i))
print('optimal percentage = ' + str(percentage_max))
w_opt, loss_opt = imp.ridge_regression(y_train,X_train,max_i)
print(compute_confusion_matrix(y_validate, X_validate.dot(w_temp)))

# # ================ Test set =============================================
# # y_test,X_test,test_labels=ld.load_csv_data("./test.csv")
# # X_test=ld.update_dataframe_median(X_test)


