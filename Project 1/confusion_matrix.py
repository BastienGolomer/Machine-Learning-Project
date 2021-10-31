import numpy as np

def compute_confusion_matrix(true_values, predicted_values):
  '''Computes a confusion matrix using numpy for two np.arrays
  true_values and predicted_values. '''

  K = len(np.unique(true_values)) # Number of classes 
  result = np.zeros((K, K))
  
  true_values=np.where(true_values>=0.5,0,1)
  predicted_values=np.where(predicted_values>=0.5,0,1)

  for i in range(len(true_values)):
    result[int(true_values[i])][int(predicted_values[i])] += 1

  return result