import numpy as np
import loading as ld
from implementations import *
import csv
from K_fold import *
import loading as ld


def fit(y, X, Ids, K):
    ''' 
    This function splits the train data into four subgroups depending on a key value of the data set : PRI_jet_num_i
    This allows to reduce the number of columns treated in the computation and obtain better, more specialised model for each subgroup.
    A K-fold is run as well for optimisation.

    Input = 
    - y = response variables
    - X = array of all the features
    - Ids = array of the identifiers of each feature
    - K = the number of subgroups required for the k-fold
    
    Output = 
    - loss_validation : the average validation loss computed
    - yhat : predicted response variable using K-fold and our split'''
    
    # the variable PRI_jet_num_i seems quite central, as many other variables depend on it
    # here the data is separated following this fact
    # construct the sets
    Ids_0, new_PRI_jet_num_0 = split_PRI_jet_num(X, Ids ,0)
    Ids_1, new_PRI_jet_num_1 = split_PRI_jet_num(X, Ids ,1)
    Ids_2, new_PRI_jet_num_2 = split_PRI_jet_num(X, Ids ,2)
    Ids_3, new_PRI_jet_num_3 = split_PRI_jet_num(X, Ids ,3)
    
    # train and compute weights using K-fold for every number of pri jet
    # the data treatment of changing the -999 by the medians is done inside run_K_fold

    [w_K_fold_0, loss_validation_0] = run_K_fold(y[Ids_0 - int(1E5)],new_PRI_jet_num_0,K)
    [w_K_fold_1, loss_validation_1] = run_K_fold(y[Ids_1 - int(1E5)],new_PRI_jet_num_1,K)
    [w_K_fold_2, loss_validation_2] = run_K_fold(y[Ids_2 - int(1E5)],new_PRI_jet_num_2,K)
    [w_K_fold_3, loss_validation_3] = run_K_fold(y[Ids_3 - int(1E5)],new_PRI_jet_num_3,K)
  
    print('loss_validation_0 = ' + str(loss_validation_0))
    print('loss_validation_1 = ' + str(loss_validation_1))
    print('loss_validation_2 = ' + str(loss_validation_2))
    print('loss_validation_3 = ' + str(loss_validation_3))

    print('Ids_0 = ' + str(Ids_0))
    print('Ids_1 = ' + str(Ids_1))
    print('Ids_2 = ' + str(Ids_2))
    print('Ids_3 = ' + str(Ids_3))


    # print('w_K_fold_0 = ' + str(w_K_fold_0))
    # print('w_K_fold_1 = ' + str(w_K_fold_1))
    # print('w_K_fold_2 = ' + str(w_K_fold_2))
    # print('w_K_fold_3 = ' + str(w_K_fold_3))


    # Aggregate the different weights we got for the PRI_jet_num values
    w_final = np.concatenate((w_K_fold_0,w_K_fold_1,w_K_fold_2,w_K_fold_3))
    # print('w_final has shape' + str(w_final.shape))
    # Ids_final = np.concatenate((Ids_0,Ids_1,Ids_2,Ids_3))
    # print('Ids_final has shape' + str(Ids_final.shape))
    # to_sort = np.vstack([w_final, Ids_final])# we stack both previous arrays in order to sort by the Id value
    # sorted = to_sort [ :, to_sort[1].argsort()]
    # w_final = sorted[0] # we keep the first line only, aka the weights

    # Compute the average validation loss we got from the K-fold
    loss_validation=1.0/4.0*(loss_validation_0 + loss_validation_1 + loss_validation_2 + loss_validation_3)

    return [w_final, loss_validation]

def split_PRI_jet_num(X, Ids, i):
    ''' 
    Inputs
    - X : array of features WITH indices to keep track of changes
    - Ids = array of the identifiers of each feature
    - i : the value of PRI_jet_num we ask for
    
    Outputs
    - PRI_jet_num_i : the features corresponding to the value i and their corresponding ids.
    
    This function splits the input Features X depending on the value i of the attribute PRI_jet_num
    In the data, i = 0,1,2,3''' 

    if (i< 0 or i> 3):
        raise ValueError("Number of jets invalid, processus killed")

    # Merge Ids and data to keep only the relevant Ids
    X = np.c_[Ids, X]
    # -8 is the position of the column PRI_jet_num

<<<<<<< HEAD
    # -8 is the position of the column PRI_jet_num
=======
>>>>>>> 3af170037ec526ffd4c27dab6b2222864b769ee0
    column_PJNum = X[:,-8] # element of the column PRI_jet_num
    PRI_jet_num_i = X[:][column_PJNum == i]

    # separate Ids and data again
    Ids = np.array(PRI_jet_num_i[:,0], dtype = 'int')
    # print('Ids of PRI_jet_num_' + str(i) + ' is ' + str(Ids))
    PRI_jet_num_i = PRI_jet_num_i[:,1:]
    # print(PRI_jet_num_i.shape)


    return Ids, PRI_jet_num_i