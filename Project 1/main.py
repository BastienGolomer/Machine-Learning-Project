import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import loading as ld
import implementations as imp
import confusion_matrix as conf
import RegressionSelection as RS

[Id, bestW] , best = RS.RegressionSelection('./train.csv')


