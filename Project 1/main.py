import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import implementations as imp
import confusion_matrix as conf
import RegressionSelection as RS
import K_fold as KF

y, X, labels = ld.load_csv_data('./train.csv')

Id = X[:,0]

indices_to_delete = np.where(X == -999)[1]
new_X = np.delete(X, indices_to_delete, axis=1)
new_X = np.delete(new_X,0,axis=1)
new_y = np.delete(y, indices_to_delete, axis=0)

labels = np.delete(labels,[0,1]) # to keep the relevant headers for the features in X


[k_fold_w, loss] = KF.K_fold(new_X, new_y)
print(loss)
# bestyhat, [Id, bestW] , best, y_validate, indices_to_delete = RS.RegressionSelection('./train.csv')
# y_test, X_test, labels = ld.load_csv_data('C:/Users/basti/Documents/EPFL/Master/MA3/ML/test.csv')
# print(y_test)
# print('here')
# new_X_test = np.delete(X_test, indices_to_delete, axis=1)
# Ids = new_X_test[:,0]
# new_X_test = np.delete(new_X_test, 0, axis=1)



# y_test_pred = new_X_test.dot(bestW[0])
# for i in range(0, len(y_test_pred)):
#     if y_test_pred[i] > 0:
#         y_test_pred[i] = 1
#     else:
#         y_test_pred[i] = -1

# # print(np.sum(y_test_pred != y_test)/len(y_test)*100)
# # ld.write_csv(Ids, y_test_pred, 'output.csv')
