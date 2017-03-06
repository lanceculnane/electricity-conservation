import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import operator
import seaborn as sns
plt.style.use('ggplot')


full_pre = pd.read_csv("combined2.csv")
to_drop = pd.read_csv('correlation_list.csv')
# full_pre.drop('school_name', 'school name', axis=1, inplace=True)
# full.fillna(0, inplace=True)
# full.head()
# full.info()

# Compute the correlation matrix
corr = full_pre.corr()


y = np.array([6.95, 10.45, 18.88, 18.97, 1.13, 5.58, 18.56, 5.16, 7.94, 6.97, 4.05, 10.28, 3.90, 7.07, 7.13, 9.73, 4.80, 5.01, 8.17, 18.57, 9.21, 13.63, 19.32, 15.98, 6.32, 4.61, 0.70, 17.94, 8.56, 16.24, 2.97, 20.88, 19.56, 8.00, 7.42, 5.29, 14.98, 15.60, 5.88, 9.87, 2.25, 10.72, 9.31, 10.97, 6.15, 8.53, 9.01, 4.47, 13.67, 6.95, 15.03, 42.05, 30.82, 6.88, 25.45, 9.15, 17.79, 1.20, 2.97, 8.26, 13.68, 7.33, 0.46, 2.63, 4.10, 10.42, 1.50]).T


def load_data(df):
    '''INPUT: pandas df
       OUTPUT: X_train, X_test, y_train, y_test
       note- NaNs are replaced with mean for that column
    '''
    # Create our imputer to replace missing values with the mean e.g.

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #axis in this case means columns even though in pandas 0 usually means rows
    imp2 = imp.fit(df) #fits it so it knows mean in each column
    full = imp2.transform(df) #transforms all NaNs to mean in each column
    X_train, X_test, y_train, y_test = train_test_split(full, y, test_size=0.3, random_state=23)
    return X_train, X_test, y_train, y_test

def run_model(X_train, X_test, y_train, y_test):
    '''runs RF model and returns RMSE'''

    rf = RandomForestRegressor(n_estimators=10,oob_score=True, max_features=70, n_jobs=3, verbose=False, min_samples_leaf=5, max_depth=7)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    return np.sqrt(mse_rf)

def create_correlation_lists(df):
    ''' takes in pandas df and spits out list of list. Each sublist contains all columns which are highly correlated'''
    # Compute the correlation matrix
    corr = df.corr()
    answer = []

    for col in range(len(corr)):
        list_of_cols = corr.ix[col,:]
        list_at_threshold = []
        for i, value in list_of_cols.iteritems():
            if value > 0.7 or value <-0.7:
                list_at_threshold.append(i)
        answer.append(list_at_threshold)

    return answer



if __name__ == '__main__':
    '''
    differences = [] # this will be my final answer... the difference between the model RMSE without a chunk of data - the best RMSE with all data.
    X_train, X_test, y_train, y_test = load_data(full_pre)
    full_rmses = []

    for i in range(10):
        full_rmses.append(run_model(X_train, X_test, y_train, y_test))
        # print full_rmses
    best_rmse = min(full_rmses)
    # print best_rmse

    lists = create_correlation_lists(full_pre)

    for i in range(143):
        new_df = full_pre.drop(lists[i],1)
        # print new_df.shape
        X_train, X_test, y_train, y_test = load_data(new_df)
        new_rmses = []

        for i in range(10):
            new_rmses.append(run_model(X_train, X_test, y_train, y_test))

        best_new_rmse = min(new_rmses)
        diff = best_new_rmse - best_rmse
        print diff
        differences.append(diff)

    print differences
    '''

    diffs = pd.read_csv('diffs_change_60.csv')
    ys = diffs['rmse_change'].values
    x = range(143)
    zipped = zip(ys, x)
    print zipped
    type(zipped)
    zipped.sort(reverse=True)
    print zipped
    y, x = zip(*zipped)

    '''
    example = [0.2, -0.1, 0.3, 0.23, -0.01, 0.31]
    x = range(6)
    new = zip(example, x)
    new.sort(reverse=True)


    y, x = zip(*new)
    # plt.bar(x, y)
    '''

    top_y = y[40:60]
    top_x = x[40:60]

    ax2 = sns.barplot(top_x,top_y, palette="Blues_d", order=top_x)
    plt.xticks(rotation=-30)
    plt.yticks(rotation=-30)
    ax2.set_ylim([0, 0.15])
    plt.show()




















'''end'''
