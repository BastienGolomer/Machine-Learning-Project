import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import implementations as imp
import confusion_matrix as conf
import RegressionSelection as RS

y, X, labels = ld.load_csv_data('./train.csv')
new_X = ld.update_dataframe_median(X)
print(X)
print(new_X)
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
