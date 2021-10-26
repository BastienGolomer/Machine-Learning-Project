import numpy as np
from loading import *

y, X, ids = load_csv_data('train.csv')

PRI_jet_num_0 = new_X[:][new_X[:,22] == 0]
new_PRI_jet_num_0 = PRI_jet_num_0.copy()
new_PRI_jet_num_0 = np.delete(new_PRI_jet_num_0, np.where(new_PRI_jet_num_0 == -999)[1], axis=1)

PRI_jet_num_1 = new_X[1:][new_X[1:,22] == 1]
new_PRI_jet_num_1 = PRI_jet_num_1.copy()
new_PRI_jet_num_1 = np.delete(new_PRI_jet_num_1, np.where(new_PRI_jet_num_1 == -999)[1], axis=1)

PRI_jet_num_2 = new_X[1:][new_X[1:,22] == 2]
new_PRI_jet_num_2 = PRI_jet_num_2.copy()
new_PRI_jet_num_2 = np.delete(new_PRI_jet_num_2, np.where(new_PRI_jet_num_2 == -999)[1], axis=1)

PRI_jet_num_3 = new_X[1:][new_X[1:,22] == 3]
new_PRI_jet_num_3 = PRI_jet_num_3.copy()
new_PRI_jet_num_3 = np.delete(new_PRI_jet_num_3, np.where(new_PRI_jet_num_3 == -999)[1], axis=1)
