import numpy as np

def compute_confusion_matrix(true_values, predicted_values):
  '''Computes a confusion matrix using numpy for two np.arrays
  true_values and predicted_values. '''

  K = len(np.unique(true_values)) # Number of classes 
  result = np.zeros((K, K))
  print(true_values)
  predicted_values[predicted_values < 0] = 0
  predicted_values[predicted_values > 0] = 1
  print(predicted_values)

  for i in range(len(true_values)):
    result[int(true_values[i])][int(predicted_values[i])] += 1

  return result