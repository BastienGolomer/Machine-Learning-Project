import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import K_fold as KF
from implementations import *
import split_fit as sf
<<<<<<< HEAD
import adding_features as af
import confusion_matrix as cm
=======
import implementations as imp
from confusion_matrix import *
>>>>>>> 3af170037ec526ffd4c27dab6b2222864b769ee0


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
#X=af.add_features(X, 2)
#X=af.build_poly(X, 2)

X_train, X_validate = np.split(new_X,[int(.7*len(X))])
y_train, y_validate = np.split(y,[int(.7*len(y))])


# ============================ might be incompatible with the next block(noKfold) ==================================================
# Now add dimensions in the form of multiplication of two columns at a time and take the one that minimizes the loss
[w_final, loss_validation] = ridge_regression(y_train, X_train,0.4)
losses=(mse(y_validate,X_validate,w_final))
#function adding columns
new_X=ld.add_dim(y,new_X,10,losses)

X_train, X_validate = np.split(new_X,[int(.7*len(X))])
y_train, y_validate = np.split(y,[int(.7*len(y))])

[w_final, loss_validation] = KF.run_K_fold(y_train, X_train,7)
y_hat_val=X_train.dot(w_final)

print(cm.compute_confusion_matrix(y_validate,y_hat_val))
print(loss_validation)

# ============================ TEST NO KFOLD ======================================================

new_X = ld.standardize_clean_dataframe(new_X)
print(new_X.max())

labels = np.delete(labels,[0,1]) # to keep the relevant headers for the features in X

# split the train.csv into a training set and a validation set
X_train, X_validate = np.split(new_X,[int(.9*len(new_X))])
y_train, y_validate = np.split(y,[int(.9*len(y))])

<<<<<<< HEAD
n = X_train.shape[1]

# w_temp, loss = imp.least_squares(y_train, X_train)
# w_temp, loss = imp.least_squares_GD(y_train,X_train,np.ones(n)/10,100,0.1)
# w_temp, loss = imp.least_squares_SGD(y_train,X_train,np.ones(n)/10,100,0.3)
# w_temp, loss = imp.ridge_regression(y_train,X_train,1.1)
w_temp, loss = imp.logistic_regression(y_train,X_train,np.ones(n)/10,100,0.1)
# w_temp, loss = imp.reg_logistic_regression(y_train,X_train,0.1,np.ones(n)/10,100,1.1)
print(compute_confusion_matrix(y_validate, X_validate.dot(w_temp)))
print(loss)


# ================ Test set =============================================
# y_test,X_test,test_labels=ld.load_csv_data("./test.csv")
# X_test=ld.update_dataframe_median(X_test)
=======
# # Now split following the PRI_jet_num, keeping the identifiers in the right order
# K = 10 # the number of subgroup desired
# w_final, loss_validation = sf.fit(y, X, Ids, K)



# ============================ TEST NO KFOLD ======================================================

new_X = ld.standardize_clean_dataframe(new_X)
print(new_X.max())

labels = np.delete(labels,[0,1]) # to keep the relevant headers for the features in X

# split the train.csv into a training set and a validation set
X_train, X_validate = np.split(new_X,[int(.9*len(new_X))])
y_train, y_validate = np.split(y,[int(.9*len(y))])

n = X_train.shape[1]

# w_temp, loss = imp.least_squares(y_train, X_train)
# w_temp, loss = imp.least_squares_GD(y_train,X_train,np.ones(n)/10,100,0.1)
# w_temp, loss = imp.least_squares_SGD(y_train,X_train,np.ones(n)/10,100,0.3)
# w_temp, loss = imp.ridge_regression(y_train,X_train,1.1)
w_temp, loss = imp.logistic_regression(y_train,X_train,np.ones(n)/10,100,0.1)
# w_temp, loss = imp.reg_logistic_regression(y_train,X_train,0.1,np.ones(n)/10,100,1.1)
print(compute_confusion_matrix(y_validate, X_validate.dot(w_temp)))
print(loss)


# ================ Test set =============================================
# y_test,X_test,test_labels=ld.load_csv_data("./test.csv")
# X_test=ld.update_dataframe_median(X_test)


>>>>>>> 3af170037ec526ffd4c27dab6b2222864b769ee0
