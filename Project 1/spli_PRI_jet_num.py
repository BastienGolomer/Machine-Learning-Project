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
    #test_x=np.c_[a,test_X]
    # Data processing : getting rid of the columns which have "-999" values
    new_X = X.copy()
    print(new_X.shape,test_X.shape)
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
    ychap1=traitements_test_set(1,test_X,PRI1[0])
    ychap2=traitements_test_set(2,test_X,PRI2[0])
    ychap3=traitements_test_set(3,test_X,PRI3[0])
    #merge the prediction and put them back in their original order
    yhat=[ychap0]
    yhat.extend(ychap1)
    yhat.extend(ychap2)
    yhat.extend(ychap3)
    for i in range (0,len(yhat[:][0][:,0])):
        yhat[:][0][i,0]=int(yhat[:][0][i,0])
        if yhat[:][0][i,1]>0:
            yhat[:][0][i,1]= 1
        else:
            yhat[:][0][i,1]= -1
    #yhat=np.array(yhat)
    print(yhat[:][0][:,0])
    y_hat_test=yhat[:][0][yhat[:][0][:,0]].sort(axis = 0)
    #compute mean loss on train
    loss=1/4*(PRI0[1]+PRI1[1]+PRI2[1]+PRI3[1])
    #permet d'avoir la moyenne des arguments mais pas reelement utile car il n'y a jamais le meme nombre d'arg et dependant de PRI_jet_num

    return loss, yhat_test


def traitements_test_set(i, test_X,w):
    PRI_test = splity_split_yo(test_X,i)
    wtest=[]
    wtest.extend(w)
    yshapoaud=PRI_test.dot(wtest)
    #print(type(PRI_test[:,0]),type(yshapoaud))
    yshapoaud=np.c_[PRI_test[:,0],yshapoaud]
    return yshapoaud
def splity_split_yo(new_X,i):
    PRI_jet_num = new_X[:][new_X[:,-8] == i]
    PRI_jet_num=ld.update_dataframe_median(PRI_jet_num)
    #new_PRI_jet_num_0 = PRI_jet_num_0.copy()
    #new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)
    return PRI_jet_num
split()