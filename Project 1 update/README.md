README:

This folder contains the first project of machine learning CS-433. In this project the objective was to classify data from the CERN between Higgs boson and not Higgs using nothing but the numpy library to classify the data. The test dataset does not have the true answer and an output file has to be submitted to "https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions/new" to have a classification score.

For more in depth detail there is a report of what methods and why they were used.

This folder should contain 6 .py files :features, gradientloss_funcions, preprocessData, proj1_helpers and run.
There sould also be a train.csv and a test.csv file, as well as an empty file to fill with the predictions called output.csv

/!\ If this output.csv is not created before nothing will be filled and there won't be an output file.

To run the project one needs to call on run.py that will execute everything if the 3 .csv files are present in the same folder.

code architecture:

All the extra .py files are imported in run.py that calls functions to do the classification.

-The features.py file contains the function to add features to the data such as polynomial terms, cross columns, or putting data through functions (trigo).

-The implementations.py file contains the 6 methods to classify asked, as well as methods to create a confusion matrix, separate the predictions in the wanted categories, batch iter, or to split the data.

-The proj1_helpers has the function to treat the csv files (read and write), as well as compute the predictions from the data and the weight vectors.

-The gradientloss_functions computes the gradients, the loss functions, as well as logistic functions.

-The preprocessData function removes columns (observations) when all of the observation (row) are equal to -999 (missing data value), changes the missing values to the median if there is only certain data that is missing and then normalise by columns the data.