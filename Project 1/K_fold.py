import numpy as np
import loading as ld
from implementations import *
from loss_functions import *


#
def K_fold_split(X, y, K):
    ''' 
    Inputs = 
    - X : array of features
    - y : array of response variables
    - K : number of subsets desired for the K-fold
    Outputs = 
    - X_train,Y_train,X_validate,Y_validate : subsets used for the k fold

    This function splits the data in datasets useable for the K-fold  
    get X and y from a train set that is not split in a validation set, as well as the parameter K to do a k-fold validation'''

    #split X and y into K pieces
    pieces=np.array_split(X, K, axis=0)
    ypieces=np.array_split(y, K)
    #creates list where each fold (case of data being split) will be stored =>(each of size K)
    X_train=[]
    Y_train=[]
    X_validate=[]
    Y_validate=[]
    for i in range (0,K):
        #the validation set consist of one of the K pieces, the append function adds a matrix element of size (lenght(X[:,0])/K,predictors)
        X_validate.append(pieces[i])
        Y_validate.append(ypieces[i])
        #putting the K-1 pieces back together in a single matrix was a bit tricky, use of np.vstack
        #also couldn't find how to initialyse the np.array to nothing so started with an array of zeros and it is ignored after the j for loop  
        temp=np.zeros(len(X[0,:]))
        ytemp=np.zeros(1)
        for j in range (0,K):
            #the if condition verifies that the validation set taken is not in the train set as well
            if j!=i:
                temp=np.vstack((temp,pieces[j]))
                ytemp=np.concatenate((ytemp,ypieces[j]), axis= None)
        #the train set for this K are added to the list
        X_train.append(temp[1:,:])
        Y_train.append(ytemp[1:]) 
    #4 list are returned, each of them of length K, and the 4 elements[k] of the lists are to be trained and validated together
    return X_train,Y_train,X_validate,Y_validate


def K_fold(X, y, K = 8):
    ''' 
    Inputs = 
    - X : array of features
    - y : array of response variables
    - K : number of subsets desired for the K-fold
    Outputs = 
    - [wtemp, loss] : weights and loss computed with the K-fold

    This function get X and y from a train set that is not split in a validation set, as well as the parameter K to do a k-fold validation
    The function calls upon K_fold_split to get data in the right shape and perform the K-fold '''

    #gets data from K_Fold_split
    trainx,trainy,valx,valy=K_fold_split(X, y, K)
    w=[]
    losses=[]
    for i in range (0,K):
        #trains the data from the ith fold to get weights, the function always return the loss (temp[1]), but it is not the one that interest us
        # as it has been used to train the data and an overfit would give a loss smaller than it will be on test 
        temp=ridge_regression(trainy[i],trainx[i],1.1)
        w.append(temp[0])
        #we compute the true loss using the ith validate set that has not been used to train the data 
        losses.append(mse(valy[i],valx[i],temp[0]))
    #compute the mean of the K losses and weight vectors that we have just computed to get the weights that have been trained on the whole dataset
    loss=1/K*sum(losses)
    w=np.array(w)
    wtemp=[]
    for i in range(0,len(w[0,:])):
        wtemp.append(1/K*sum(w[:,i]))
    #return the loss and weight vectors that have been traine on the FULL X that we had at start.
    return [wtemp, loss]

#ridge :0.16899386241189956 gamma=1.1
#least square 0.17846086928048066
#logistic_regression ~2.07 200 iter et gamma~0.1
#reg_log gamma 0.00001 lambda_ 0.4 iter 50

#run K fold directly from this file    
def run_K_fold(K):
    '''Input = K : number of subsets asked for the K-fold
    Output = creates a csv file using write_csv file (see the documentation on the latter) 
    
    This function performs a K-fold on the train test to directly validate on the test set.
    Before it calls functions K_fold, function update_dataframe_median is used to clean the data (see the documentatio of the latter)
    '''
    #load data
    y,X,ids=ld.load_csv_data("./train.csv")
    test_y,test_X,test_ids=ld.load_csv_data("./test.csv")
    #data treatment
    X=ld.update_dataframe_median(X)
    y=ld.update_dataframe_median(y)
    test_X=ld.update_dataframe_median(test_X)
    #compute weights
    w=K_fold(X,y,K)
    #make predictions
    y_hat=test_X.dot(w[0])
    #output the prediction
    ld.write_csv(test_X[:,0],y_hat,'output_ridge.csv')
    return
run_K_fold(15)