# the following is from the excellent tutorial by Jason Brownlee- he's got a lot of great ones, and are easier to understand than most!: http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


# #note! first you have to download the housing data as housing.csv from here:
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data

#I'm altering the original toy data and code for my data w/ electricity
# This NN is set up as a Regressor. To change to classifier, simply put "Classifier" anywhere you see "Regressor"
# This NN is set up to do StandardScaling for you - mean and std dev are zeroed out
import datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD



# # load dataset
# dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
# dataset = dataframe.values
# # split into input (X) and output (Y) variables
# X = dataset[:,0:13]
# Y = dataset[:,13]
#

def load_data():
    ''' loads and reshapes data '''
    y = np.array([6.95, 10.45, 18.88, 18.97, 1.13, 5.58, 18.56, 5.16, 7.94, 6.97, 4.05, 10.28, 3.90, 7.07, 7.13, 9.73, 4.80, 5.01, 8.17, 18.57, 9.21, 13.63, 19.32, 15.98, 6.32, 4.61, 0.70, 17.94, 8.56, 16.24, 2.97, 20.88, 19.56, 8.00, 7.42, 5.29, 14.98, 15.60, 5.88, 9.87, 2.25, 10.72, 9.31, 10.97, 6.15, 8.53, 9.01, 4.47, 13.67, 6.95, 15.03, 42.05, 30.82, 6.88, 25.45, 9.15, 17.79, 1.20, 2.97, 8.26, 13.68, 7.33, 0.46, 2.63, 4.10, 10.42, 1.50]).T
    # print type(y)
    # print y
    # print y.shape
    # preX = np.genfromtxt('combined.csv', delimiter=",")
    # preX = preX[1:]
    new = pd.read_csv('combined.csv')
    new.fillna(0, inplace=True)
    preX = new.values
    # preX = preX.T
    # print preX.shape
    # print type(preX)
    # print preX
    # ok good! So it looks like both y and preX are both successfully loaded
    # and they both have 67 rows and 67 answers (y) and the preX has 1440 columns
    X_train, X_test, y_train, y_test = train_test_split(preX, y, test_size=0.25, random_state=23)

    return X_train, y_train, X_test, y_test



# define base mode
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(1440, input_dim=157, init='uniform', activation='tanh'))
	#changed dense from 13 to 1440 since they had 13 columns (features) which
	# was used to predict housing prices, whereas I am slopily attempting to
	# use ever 15-minute electricity usage as a column...
	model.add(Dense(720, init='uniform', activation='tanh'))
	model.add(Dense(360, init='uniform', activation='tanh'))
	model.add(Dense(180, init='uniform', activation='tanh'))
	# model.add(Dense(90, input_dim=180, init='zero', activation='tanh'))
	# model.add(Dense(45, input_dim=90, init='zero', activation='tanh'))
	# model.add(Dense(22, input_dim=45, init='zero', activation='tanh'))
	# model.add(Dense(11, input_dim=22, init='zero', activation='tanh'))
	# model.add(Dense(4, input_dim=11, init='zero', activation='tanh'))
	# model.add(Dense(2, input_dim=4, init='zero', activation='tanh'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	sgd = SGD(lr=0.001, decay=1e-7, momentum=.5) # using stochastic gradient descent (keep)
	model.compile(loss='mean_squared_error', optimizer='adam') # optimizer='adam' works better many times
	return model

#run model
def train_and_evaluate_NN(X_train, y_train, X_test, y_test, verbosity=1, test=True, y_pred=False):
    '''INPUT: X_train, y_train, X_test, y_test
       INPUT: verbosity=1: if left on 1 it will run a little slower but show progress bars. Set verbosity=0 and it will get rid of progress bars
       INPUT: test=True: This will train the NN on X_train, y_train, and then if set to True, it will also run on X_test, y_test and give metric results on that unseen data
       INPUT: y_pred=True will print all y-predicted values. NOTE! This doubles the processing time on the test data portion
       OUTPUT: Results of X_train, y_train, and if test=True, also results of X_test, y_test
       OUTPUT: y_pred if set to True
       '''


    # fix random seed for reproducibility
    seed = 23
    np.random.seed(seed)
    # # evaluate model with standardized dataset
    # seed = 23
    # numpy.random.seed(seed)
    # # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

    # evaluate model with standardized dataset

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=10, verbose=verbosity)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=9, random_state=seed)
    results = cross_val_score(pipeline, X_train, y_train, cv=kfold, verbose=0)
    print("Avg (std) of Training Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    # estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=10, verbose=1)

    if test:
        test_results = cross_val_score(pipeline, X_test, y_test, verbose=0)
        print "Test avg MSE (and std) Results: ", test_results.mean(), test_results.std()

    if y_pred:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print "y_pred: ", y_pred
        return y_pred


    # kfold = KFold(n_splits=9, random_state=seed)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))





if __name__ == '__main__':
    start = datetime.datetime.now().time()
    X_train, y_train, X_test, y_test = load_data()
    y_pred = train_and_evaluate_NN(X_train, y_train,X_test, y_test)
    print "Time when we started: ", start
    print "Time when we finished: ", datetime.datetime.now().time()
