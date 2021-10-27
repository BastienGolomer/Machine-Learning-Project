import numpy as np
import loading as ld

def K_fold(K, train, y):
    len_valid=len(train)/K
    print(y)
    pieces=np.array_split(train, K, axis=0)
    ypieces=np.array_split(y, K)
    X_train=np.zeros([2,len(train[0,:])])
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
        X_train=np.r_[X_train,temp[1:,:]]
        Y_train=np.append(Y_train,ytemp[1:])    
    return X_train[2:,:],Y_train,X_validate,Y_validate

def main():
    y, X, labels, ids = ld.load_csv_data('./train.csv')
    trainx,trainy,valx,valy=K_fold(8,X[1:5000,:],y[1:5000])
    print (len(trainx))
    print (trainy)
        
    
main()