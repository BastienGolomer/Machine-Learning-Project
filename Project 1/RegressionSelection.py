import numpy as np
import loading as ld
import implementations as imp


def RegressionSelection(csv_filepath) :
    methods = ['LeastSquare', 'LeastSquareGD', 'LeastSquareSGD', 'RidgeRegression', 'LogisticRegression', 'RegLogisticRegression'  ]
    evals = ['imp.least_squares(y_train,X_train)', 'imp.least_square_GD(y_train,X_train,np.ones(30),100,0.8)', 'imp.least_squares_SGD(y_train,X_train,np.ones(30),100,0.3)', 
             'imp.ridge_regression(y_train,X_train,0.5)', 'imp.logistic_regression(y_train,X_train,np.ones(30),100,0.1)', 'imp.reg_logistic_regression(y_train,X_train,0.1,np.ones(30),100,0.1)'   ]


    # load the data using our own method + Data processing Training Set: getting rid of the columns which have "-999" values
    y, X, labels = ld.load_csv_data(csv_filepath)
    indices_to_delete = np.where(X == -999)[1]
    new_X = np.delete(X, indices_to_delete, axis=1)
    new_y = np.delete(y, indices_to_delete, axis=0)

    Id = new_X[:,0]
    np.delete(labels,[0,1]) # to keep the relevant headers for the features in X

    # split the train.csv into a training set and a validation set
    X_train, X_validate = np.split(new_X,[int(.8*len(new_X))])
    y_train, y_validate = np.split(y,[int(.8*len(y))])

    best = 'LeastSquare'
    worstcount = 6e6
    best_yhat = []
    bestW = []

    for j in range(len(methods)-1):

        w=eval(evals[j])
        yhat=X_validate.dot(w[0])
        count= 0
        for i in range (0,len(yhat)):
            if yhat[i]>0:
                yhat[i]= 1
            else:
                yhat[i]= -1

            if yhat[i] != y_validate[i] :
                count += 1    

        if count < worstcount :
            worstcount = count
            bestyhat = yhat
            bestW = w
            best = methods[j]


# we return the best weights we could find for this dataset and the method used
    return [Id, bestW] , best



