import numpy as np
import loading as ld
from implementations import *
import csv
from K_fold import *
import loading as ld


def split():
    y,X,ids=ld.load_csv_data("./train.csv")
    test_y,test_X,test_ids=ld.load_csv_data("./test.csv")
    X=np.c_[y,X]
    a=np.linspace(1,len(test_X[:,1]),len(test_X[:,1]))
    test_x=np.c_[a,test_X]
    # Data processing : getting rid of the columns which have "-999" values
    new_X = X.copy()
    # the variable PRI_jet_num seems quite central, as many other variables depend on it
    # here the data is separated following this fact
    #train
    new_PRI_jet_num_0 = splity_split_yo(new_X,0)
    new_PRI_jet_num_1 = splity_split_yo(new_X,1)
    new_PRI_jet_num_2 = splity_split_yo(new_X,2)
    new_PRI_jet_num_3 = splity_split_yo(new_X,3)
    
    #print(new_PRI_jet_num_0[:,0].shape, new_PRI_jet_num_0[:,1:].shape)
    PRI0=K_fold(new_PRI_jet_num_0[:,0],new_PRI_jet_num_0[:,1:])
    PRI1=K_fold(new_PRI_jet_num_1[:,0],new_PRI_jet_num_1[:,1:])
    PRI2=K_fold(new_PRI_jet_num_2[:,0],new_PRI_jet_num_2[:,1:])
    PRI3=K_fold(new_PRI_jet_num_3[:,0],new_PRI_jet_num_3[:,1:])
    #split for test
    ychap0=traitements_test_set(0,test_X,PRI0[0])
    ychap1=traitements_test_set(1,test_X,PRI0[0])
    ychap2=traitements_test_set(2,test_X,PRI0[0])
    ychap3=traitements_test_set(3,test_X,PRI0[0])
    
    yhat=[ychap0]
    yhat.extend(ychap1)
    yhat.extend(ychap2)
    yhat.extend(ychap3)
    #print(PRI0)
    loss=1/4*(PRI0[1]+PRI1[1]+PRI2[1]+PRI3[1])
    #permet d'avoir la moyenne des arguments mais pas reelement utile car il n'y a jamais le meme nombre d'arg
    w=[PRI0[0]]
    w.append(PRI1[0])
    w.append(PRI2[0])
    w.append(PRI3[0])
    wtemp=[]
    for i in range(0,len(w[0])):
        wtemp.append(1/4*(w[0][i]+w[1][i]+w[2][i]+w[3][i]))
    print (loss)
    return wtemp,loss


def traitements_test_set(i, test_X,w):
    PRI_test = splity_split_yo(test_X,i)
    wtest=[1]
    wtest.extend(w)
    yshapoaud=PRI_test.dot(wtest)
    yshapoaud=np.c_(PRI_test[:,0],yshapoaud)
    return yshapoaud
def splity_split_yo(new_X,i):
    PRI_jet_num_0 = new_X[:][new_X[:,24] == i]
    new_PRI_jet_num_0 = PRI_jet_num_0.copy()
    new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)
    return new_PRI_jet_num_0
split()