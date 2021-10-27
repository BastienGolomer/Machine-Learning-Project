import numpy as np
import loading as ld
from implementations import *
import csv
from K_fold import *
import loading as ld


def split():
    y,X,ids=ld.load_csv_data("./train.csv")
    X=np.c_(y,X)
    # Data processing : getting rid of the columns which have "-999" values
    new_X = X.copy()
    #new_X = np.delete(new_X, np.where(new_X == -999)[1], axis=1)
    print(X)
    # the variable PRI_jet_num seems quite central, as many other variables depend on it
    # here the data is separated following this fact
    PRI_jet_num_0 = new_X[:][new_X[:,24] == 0]
    new_PRI_jet_num_0 = PRI_jet_num_0.copy()
    new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)
    print(new_PRI_jet_num_0)
    PRI0=K_fold(new_PRI_jet_num_0[:,1],new_PRI_jet_num_0[:,2:])
    
    PRI_jet_num_1 = new_X[1:][new_X[1:,25] == 1]
    new_PRI_jet_num_1 = PRI_jet_num_1.copy()
    new_PRI_jet_num_1 = np.delete(new_PRI_jet_num_1, np.where(new_PRI_jet_num_1 == -999)[1], axis=1)
    PRI1=K_fold(new_PRI_jet_num_1[:,1],new_PRI_jet_num_1[:,2:])
    
    PRI_jet_num_2 = new_X[1:][new_X[1:,25] == 2]
    new_PRI_jet_num_2 = PRI_jet_num_2.copy()
    new_PRI_jet_num_2 = np.delete(new_PRI_jet_num_2, np.where(new_PRI_jet_num_2 == -999)[1], axis=1)
    PRI2=K_fold(new_PRI_jet_num_2[:,1],new_PRI_jet_num_2[:,2:])
    
    PRI_jet_num_3 = new_X[1:][new_X[1:,25] == 3]
    new_PRI_jet_num_3 = PRI_jet_num_3.copy()
    new_PRI_jet_num_3 = np.delete(new_PRI_jet_num_3, np.where(new_PRI_jet_num_3 == -999)[1], axis=1)
    PRI3=K_fold(new_PRI_jet_num_3[:,1],new_PRI_jet_num_3[:,2:])
    
    loss=1/8*sum(PRI0[1]+PRI1[1]+PRI2[1]+PRI3[1])
    #print ((w[0]))
    w=np.array(PRI0[0],PRI1[0],PRI2[0],PRI3[0])
    wtemp=[]
    for i in range(0,len(w[0,:])):
        wtemp.append(1/8*sum(w[:,i]))
    print(wtemp, loss)
    return wtemp,loss
split()