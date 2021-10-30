import numpy as np

def compute_confusion_matrix(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred. '''

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))
  true=np.where(true<0,0,1)
  pred=np.where(pred<0,0,1)
  for i in range(len(true)):
    result[int(true[i])][int(pred[i])] += 1

  return result