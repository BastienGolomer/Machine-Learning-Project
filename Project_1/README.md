README:

This folder contains the first project of machine learning CS-433. In this project the objective is to classify data from the CERN between Higgs boson and not Higgs using only the numpy library to classify the data as well as a training dataset. The test dataset does not have the true answer and an output file has to be submitted to "https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions/new" to have a classification score.

For more in depth details on what methods were used and why, check the report.pdf

This folder should contain two subfolders : scripts and data. The former contains 6 .py files :features, gradientloss_funcions, preprocessData, proj1_helpers and run. The latter contains 3 .csv files : train.csv, test.csv file, as well as an __empty file: output.csv__ to fill with the predictions.

/!\ If this output.csv is not created before running the code nothing will be filled and there won't be an output file. If the output.csv file is missing, please create it.

To run the project one needs to __call run.py__ (using python) that will execute everything if the 3 .csv files are present __in the data folder__ (otherwise the paths will be compromised). Make sure your Python interpreter has version 3.8 or above.

Code architecture:

* All the extra .py files are imported in run.py that calls functions to do the classification.

* The features.py file contains the function to add features to the data such as polynomial terms, cross columns, or putting data through functions (trigo).

* The implementations.py file contains the 6 methods to classify asked, as well as methods to create a confusion matrix, separate the predictions in the wanted categories, batch iter, or to split the data.

* The proj1_helpers has the function to treat the csv files (read and write), as well as compute the predictions from the data and the weight vectors.

* The gradientloss_functions computes the gradients, the loss functions, as well as logistic functions.

* The preprocessData function changes the missing values to the median of the feature and then normalise by features the data.
