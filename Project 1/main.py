import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import K_fold as KF
import split_fit as sf

# methods that can be used as regressions
methods = ['LeastSquare', 'LeastSquareGD', 'LeastSquareSGD', 'RidgeRegression', 'LogisticRegression', 'RegLogisticRegression'  ]

#load data
y,X,labels =ld.load_csv_data("./train.csv")
y_test,X_test,test_labels=ld.load_csv_data("./test.csv")
X_test=ld.update_dataframe_median(X_test)

# isolate the identifiers of each feature
Ids = X[:,0]

# remove the Ids from X
new_X = np.delete(X,0,axis=1)

# remove the lables of id and y from the "labels", to keep the relevant headers for the features in X
labels = np.delete(labels,[0,1]) 

# Now split following the PRI_jet_num, keeping the identifiers in the right order
K = 10 # the number of subgroup desired
[w_final, loss_validation] = sf.fit(y, X, Ids, K)


# make predictions
y_hat_test=X_test.dot(w_final)

# output the prediction
A=Ids
B=np.where(y_hat_test<0,-1,1)
ld.write_csv(A,B,'output_test.csv')