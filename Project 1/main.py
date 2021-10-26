import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import implementations as imp
import confusion_matrix as conf

ids_csv = np.linspace(35000,568240, 218240)


# load the data using our own method
y_train, X_train, ids_train = ld.load_csv_data('./train.csv')
np.delete(ids_train,[0,1]) # to keep the relevant headers for the features in X
y_test, X_test, ids_test = ld.load_csv_data('./train.csv')
np.delete(ids_test,[0,1]) # to keep the relevant headers for the features in X


# Data processing Training Set: getting rid of the columns which have "-999" values
new_X = X_train.copy()
new_X = np.delete(new_X, np.where(new_X == -999)[1], axis=1)

# #split the train.csv into a training set and a validation set
# X_train, X_validate = np.split(new_X,[int(.5*len(X_train))])
# y_train, y_validate = np.split(y_train,[int(.5*len(y_train))])


# Data processing Test Set : getting rid of the columns which have "-999" values
new_X = X_test.copy()
new_X = np.delete(new_X, np.where(new_X == -999)[1], axis=1)

# least squares
w_ls=imp.least_squares(y_train,X_train)
yhat=X_test.dot(w_ls[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1
ld.write_csv(ids_csv, yhat, 'LeastSquare.csv')





# least squares Gradient Descent
w_lsgd=imp.least_square_GD(y_train,X_train,np.ones(30),100,0.8)
yhat=X_test.dot(w_lsgd[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1
ld.write_csv(ids_csv, yhat, 'LeastSquareGD.csv')



# least squares stochastic Gradient Descent
w_lssgd=imp.least_squares_SGD(y_train,X_train,np.ones(30),100,0.3)
yhat=X_test.dot(w_lssgd[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1
ld.write_csv(ids_csv, yhat, 'LeastSquareSGD.csv')


# ridge regression
w_rr=imp.ridge_regression(y_train,X_train,0.5)
yhat=X_test.dot(w_rr[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1
ld.write_csv(ids_csv, yhat, 'RidgeRegression.csv')


# logistic regression
w_lr=imp.logistic_regression(y_train,X_train,np.ones(30),100,0.1)
yhat=X_test.dot(w_lr[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1
ld.write_csv(ids_csv, yhat, 'LogisticRegression.csv')


# regularised locistic regression
w_rlr=imp.reg_logistic_regression(y_train,X_train,0.1,np.ones(30),100,0.1)
yhat=X_test.dot(w_rlr[0])
for i in range (0,len(yhat)):
    if yhat[i]>0:
        yhat[i]= 1
    else:
         yhat[i]= -1

ld.write_csv(ids_csv, yhat, 'RegLogisticRegression.csv')
