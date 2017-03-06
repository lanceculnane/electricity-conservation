# based on feature importance, I'm now only going to take the top 20 columns

import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
plt.style.use('ggplot')


full_pre = pd.read_csv("combined2.csv")
# to_drop = pd.read_csv('correlation_list.csv')
# full_pre.drop('school_name', 'school name', axis=1, inplace=True)
# full.fillna(0, inplace=True)
# full.head()
# full.info()

# Compute the correlation matrix
# corr = full_pre.corr()


y = np.array([6.95, 10.45, 18.88, 18.97, 1.13, 5.58, 18.56, 5.16, 7.94, 6.97, 4.05, 10.28, 3.90, 7.07, 7.13, 9.73, 4.80, 5.01, 8.17, 18.57, 9.21, 13.63, 19.32, 15.98, 6.32, 4.61, 0.70, 17.94, 8.56, 16.24, 2.97, 20.88, 19.56, 8.00, 7.42, 5.29, 14.98, 15.60, 5.88, 9.87, 2.25, 10.72, 9.31, 10.97, 6.15, 8.53, 9.01, 4.47, 13.67, 6.95, 15.03, 42.05, 30.82, 6.88, 25.45, 9.15, 17.79, 1.20, 2.97, 8.26, 13.68, 7.33, 0.46, 2.63, 4.10, 10.42, 1.50]).T


def get_best_cols(df):
    '''Takes in pandas df and spits out only the best rows (I've hard-coded it for now)'''
    best = df.ix[:,[85, 111, 16, 81, 142, 82, 116, 129, 20, 120, 95, 117, 123, 127, 30, 33, 119, 84, 86, 130]]
    return best


def load_data(df):
    '''INPUT: pandas df
       OUTPUT: X_train, X_test, y_train, y_test
       note- NaNs are replaced with mean for that column
    '''
    # Create our imputer to replace missing values with the mean e.g.

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #axis in this case means columns even though in pandas 0 usually means rows
    imp2 = imp.fit(df) #fits it so it knows mean in each column
    full = imp2.transform(df) #transforms all NaNs to mean in each column
    X_train, X_test, y_train, y_test = train_test_split(full, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

def run_model(X_train, X_test, y_train, y_test):
    '''runs RF model and returns RMSE'''

    rf = RandomForestRegressor(n_estimators=1000,oob_score=True, max_features=10, n_jobs=3, verbose=False, min_samples_leaf=5, max_depth=7)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    return np.sqrt(mse_rf)



if __name__ == '__main__':
    best = get_best_cols(full_pre)
    X_train, X_test, y_train, y_test = load_data(best)
    X_train2, X_test2, y_train2, y_test2 = load_data(full_pre)
    print 'best 20: ', run_model(X_train, X_test, y_train, y_test)
    print 'all 143: ', run_model(X_train2, X_test2, y_train2, y_test2)













    '''end'''
