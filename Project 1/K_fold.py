import numpy as np
import loading as ld
from implementations import *
from loss_functions import *
from confusion_matrix import *


#
def K_fold_split(y, X_train,  K):
    ''' 
    Inputs = 
    - X_train : array of features
    - y : array of response variables
    - K : number of subsets desired for the K-fold
    Outputs = 
    - X_train,Y_train,X_validate,Y_validate : subsets used for the k fold

    This function splits the data in datasets useable for the K-fold  
    get X_train and y from a train set that is not split in a validation set, as well as the parameter K to do a k-fold validation'''

    #split X_train and y into K pieces
    pieces=np.array_split(X_train, K, axis=0)
    ypieces=np.array_split(y, K)

    #creates list where each fold (case of data being split) will be stored =>(each of size K)
    X_train_split=[]
    Y_train_split=[]
    X_validate=[]
    Y_validate=[]

    for i in range (0,K):
        #the validation set consist of one of the K pieces, the append function adds a matrix element of size (lenght(X_train[:,0])/K,predictors)
        X_validate.append(pieces[i])
        Y_validate.append(ypieces[i])
        #putting the K-1 pieces back together in a single matrix was a bit tricky, use of np.vstack
        #also couldn't find how to initialise the np.array to nothing so started with an array of zeros and it is ignored after the j for loop  
        w_temp=np.zeros(pieces[i].shape[1])
        ytemp=np.zeros(1)
        
        for j in range (0,K):
            #the if condition verifies that the validation set taken is not in the train set as well
            if j!=i:
                w_temp=np.vstack((w_temp,pieces[j]))
                ytemp=np.concatenate((ytemp,ypieces[j]), axis= None)

        #the train set for this K are added to the list
        X_train_split.append(w_temp[1:,:])
        Y_train_split.append(ytemp[1:]) 
    #4 list are returned, each of them of length K, and the 4 elements[k] of the lists are to be trained and validated together
    return X_train_split,Y_train_split,X_validate,Y_validate


def K_fold( y, X_train, K = 10):
    ''' 
    Inputs = 
    - X_train : array of features
    - y : array of response variables
    - K : number of subsets desired for the K-fold
    Outputs = 
    - [wtemp, loss] : weights and loss computed with the K-fold

    This function get X_train and y from a train set that is not split in a validation set, as well as the parameter K to do a k-fold validation
    The function calls upon K_fold_split to get data in the right shape and perform the K-fold '''

    #gets data from K_Fold_split
    trainx,trainy,valx,valy=K_fold_split(y, X_train, K)

    # prepare storage
    w_K_fold_i=[]
    losses=[]

    for i in range (0,K):
        #trains the data from the ith fold to get weights, the function always return the loss (w_temp[1]), but it is not the one that interest us
        # as it has been used to train the data and an overfit would give a loss smaller than it will be on test 
        
        size_w = trainx[i].shape[1]
<<<<<<< HEAD
        # w_temp=ridge_regression(trainy[i],trainx[i],1.1)
        # w_temp = least_squares_SGD(trainy[i],trainx[i], np.random.rand(trainx[i].shape[1]),100,0.1)
        [w_temp, loss] = reg_logistic_regression(trainy[i],trainx[i],0.1,np.random.rand(size_w),100,0.0001)
=======
>>>>>>> 3af170037ec526ffd4c27dab6b2222864b769ee0

        # [w_temp, loss] = least_squares(trainy[i], trainx[i])
        [w_temp, loss]=ridge_regression(trainy[i],trainx[i],1.1)
        # [w_temp, loss] = least_squares_SGD(trainy[i],trainx[i], np.random.rand(trainx[i].shape[1]),100,0.1)
        # [w_temp, loss] = reg_logistic_regression(trainy[i],trainx[i],0.1,np.random.rand(size_w),100,0.1)

        w_K_fold_i.append(w_temp)

        # we compute the true loss using the ith validate set that has not been used to train the data

        losses.append(mse(valy[i],valx[i],w_temp))

    #compute the mean of the K losses and weight vectors that we have just computed to get the weights that have been trained on the whole dataset
    loss_validation=1/K*sum(losses)
    w_K_fold_i=np.array(w_K_fold_i)

    w_K_fold =[]
    for d in range(0,size_w):
        w_K_fold.append(1/K*sum(w_K_fold_i[:,d]))
    #return the loss and weight vectors that have been trained on the FULL X_train that we had at start.
    return w_K_fold, loss_validation, valx, valy

#ridge :0.16899386241189956 gamma=1.1
#least square 0.17846086928048066
#logistic_regression ~2.07 200 iter et gamma~0.1
#reg_log gamma 0.00001 lambda_ 0.4 iter 50

#run K fold directly from this file    
def run_K_fold(y,X_train, K):
    '''
    Input = K : number of subsets asked for the K-fold
    Output = creates a csv file using write_csv file (see the documentation of the latter) 
    
    This function performs a K-fold on the train test to directly validate on the test set.
    Before it calls functions K_fold, function update_dataframe_median is used to clean the data (see the documentatio of the latter)
    '''

    # data treatment, see documentation on update_dataframe_median
    X_train=ld.update_dataframe_median(X_train)


    #compute weights
    w_K_fold, loss_validation, y_validation, X_validation=K_fold(y, X_train,K)

<<<<<<< HEAD
    # # print the confusion matrix
    print (compute_confusion_matrix(y, X_train[:,1:].dot(w_K_fold[0])))
=======
    # print the confusion matrix
    print (compute_confusion_matrix(y_validation, X_validation.dot(w_K_fold)))
>>>>>>> 3af170037ec526ffd4c27dab6b2222864b769ee0

    return w_K_fold, loss_validation
