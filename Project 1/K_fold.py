import numpy as np
import loading as ld
from implementations import *
from loss_functions import *

def K_fold_split(K, train, y):
    len_valid=len(train)/K
    pieces=np.array_split(train, K, axis=0)
    ypieces=np.array_split(y, K)
    X_train=[]
    Y_train=[]
    X_validate=[]
    Y_validate=[]
    for i in range (0,K):
        X_validate.append(pieces[i])
        Y_validate.append(ypieces[i])
        temp=np.zeros(len(train[0,:]))
        ytemp=np.zeros(1)
        for j in range (0,K):
            if j!=i:
                temp=np.vstack((temp,pieces[j]))
                ytemp=np.concatenate((ytemp,ypieces[j]), axis= None)
        X_train.append(temp[1:,:])
        Y_train.append(ytemp[1:]) 
    return X_train,Y_train,X_validate,Y_validate

def K_fold(y,X):
    trainx,trainy,valx,valy=K_fold_split(8,X,y)
    w=[]
    losses=[]
    #valx=np.array(valx)
    for i in range (0,8):
        temp=ridge_regression(trainy[i],trainx[i],0.4)
        w.append(temp[0])
        losses.append(mse(valy[i],valx[i],temp[0]))
    loss=1/8*sum(losses)
    w=np.array(w)
    wtemp=[]
    for i in range(0,len(w[0,:])):
        wtemp.append(1/8*sum(w[:,i]))
    return [wtemp, loss]

    
