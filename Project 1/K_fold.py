import numpy as np
import loading as ld
from implementations import *

def K_fold_split(K, train, y):
    len_valid=len(train)/K
    #print(y)
    pieces=np.array_split(train, K, axis=0)
    ypieces=np.array_split(y, K)
    X_train=[]
    Y_train=[]
    X_validate=[]
    Y_validate=[]
    for i in range (0,K):
        np.append(X_validate,pieces[i])
        np.append(Y_validate,ypieces[i])
        temp=np.zeros(len(train[0,:]))
        #print (len(temp))
        ytemp=np.zeros(1)
        for j in range (0,K):
            if j!=i:
                temp=np.vstack((temp,pieces[j]))
                ytemp=np.concatenate((ytemp,ypieces[j]), axis= None)
        #print(temp[1,:])
        #print(X_train)
        X_train.append(temp[1:,:])
        Y_train.append(ytemp[1:])    
    return X_train,Y_train,X_validate,Y_validate

def K_fold():
    y, X, labels = ld.load_csv_data('./train.csv')
    trainx,trainy,valx,valy=K_fold_split(8,X[1:5000,:],y[1:5000])

    w=[]
    losses=[]
    for i in range (0,8):
        temp=least_squares(trainy[i],trainx[i])
        #print(trainx)
        w.append(temp[0])
        losses.append(temp[1])
    loss=1/8*sum(losses)
    #print ((w[0]))
    w=np.array(w)
    wtemp=[]
    for i in range(0,len(w[0,:])):
        wtemp.append(1/8*sum(w[:,i]))
    print(wtemp)
    return wtemp, loss

    
