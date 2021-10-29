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
    # the variable PRI_jet_num seems quite central, as many other variables depend on it
    # here the data is separated following this fact
    #train
    new_PRI_jet_num_0 = splity_split_yo(new_X,0)
    new_PRI_jet_num_1 = splity_split_yo(new_X,1)
    new_PRI_jet_num_2 = splity_split_yo(new_X,2)
    new_PRI_jet_num_3 = splity_split_yo(new_X,3)
    
    #print(new_PRI_jet_num_0[:,0].shape, new_PRI_jet_num_0[:,1:].shape)
    PRI0=K_fold(new_PRI_jet_num_0[:,0],new_PRI_jet_num_0[:,1:])
    print (X,y,"xy ")
    PRI1=K_fold(new_PRI_jet_num_1[:,0],new_PRI_jet_num_1[:,1:])
    PRI2=K_fold(new_PRI_jet_num_2[:,0],new_PRI_jet_num_2[:,1:])
    PRI3=K_fold(new_PRI_jet_num_3[:,0],new_PRI_jet_num_3[:,1:])
    #split for test
    ychap0=traitements_test_set(0,test_X,PRI0[0])
    ychap1=traitements_test_set(1,test_X,PRI1[0])
    ychap2=traitements_test_set(2,test_X,PRI2[0])
    ychap3=traitements_test_set(3,test_X,PRI3[0])
    #merge the prediction
    yhat=list(ychap0)
    yhat.extend(ychap1)
    yhat.extend(ychap2)
    yhat.extend(ychap3)
    yhat=np.array(yhat)
    #get the predictions in the right format and in the right order
    A=yhat[:,0].astype(int)
    print(yhat[:,1])
    B=np.where(yhat[:,1]<0,-1,1)
    yhat=np.array([A,B])
    yhat=yhat.T
    yhat_test=yhat[yhat[:,0].argsort(0)]
    #compute mean loss on train
    loss=1/4*(PRI0[1]+PRI1[1]+PRI2[1]+PRI3[1])
    print(loss)
    ld.write_csv(yhat_test[:,0], yhat_test[:,1],'split_PRI_jet_num.csv')
    return loss, yhat_test


def traitements_test_set(i, test_X,w):
    PRI_test = splity_split_yo(test_X,i)
    wtest=[]
    wtest.extend(w)
    yshapoaud=PRI_test.dot(wtest)
    yshapoaud=np.c_[PRI_test[:,0],yshapoaud]
    return yshapoaud


def splity_split_yo(new_X,i):
    PRI_jet_num = new_X[:][new_X[:,-8] == i]
    PRI_jet_num=ld.update_dataframe_median(PRI_jet_num)
    PRI_jet_num=ld.standardize_mat(PRI_jet_num)
    #new_PRI_jet_num_0 = PRI_jet_num_0.copy()
    #new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)
    return PRI_jet_num
split()