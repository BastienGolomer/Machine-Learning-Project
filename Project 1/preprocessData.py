import numpy as np
from loading import *

def cleanDataJetNum(path_data):
  # Load the dataset we wish to clean
  y, X, _ = load_csv_data(path_data)
  
  '''We noticed that the variable PRI_jet_num affects a lot a certain number of other columns, hence we decide to clean the dataset
     according to this variable. We delete columns for which the value of PRI_jet_num affects them, creating the -999 values'''

  PRI_jet_num_0 = new_X[:][new_X[:,23] == 0]
  new_PRI_jet_num_0 = PRI_jet_num_0.copy()
  new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)

  PRI_jet_num_1 = new_X[1:][new_X[1:,23] == 1]
  new_PRI_jet_num_1 = PRI_jet_num_1.copy()
  new_PRI_jet_num_1 = np.delete(new_PRI_jet_num_1, np.where(new_PRI_jet_num_1 == -999)[1], axis=1)

  PRI_jet_num_2 = new_X[1:][new_X[1:,23] == 2]
  new_PRI_jet_num_2 = PRI_jet_num_2.copy()
  new_PRI_jet_num_2 = np.delete(new_PRI_jet_num_2, np.where(new_PRI_jet_num_2 == -999)[1], axis=1)

  PRI_jet_num_3 = new_X[1:][new_X[1:,23] == 3]
  new_PRI_jet_num_3 = PRI_jet_num_3.copy()
  new_PRI_jet_num_3 = np.delete(new_PRI_jet_num_3, np.where(new_PRI_jet_num_3 == -999)[1], axis=1)
  
  return new_PRI_jet_num0, new_PRI_jet_num1, new_PRI_jet_num2, new_PRI_jet_num3
