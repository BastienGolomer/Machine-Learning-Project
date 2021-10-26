import numpy as np
import csv

with open("train.csv",'r') as custfile:
rows=csv.reader(custfile,delimiter=',')
for r in rows:
print(r)train= csv.read(train)

def plot_categories():
    for i in train [i,:]:
        plot (i, train[1]) 
     
def plot_pairwise():
for i in train [i,:]:
    for j in train [j,:]:    
        plot (i,j)