import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from string import letters



full_pre = pd.read_csv("combined.csv")
# full_pre.drop('school_name', 'school name', axis=1, inplace=True)
# full.fillna(0, inplace=True)
# full.head()
# full.info()

# Create our imputer to replace missing values with the mean e.g.

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #axis in this case means columns even though in pandas 0 usually means rows



imp2 = imp.fit(full_pre) #fits it so it knows mean in each column
full = imp2.transform(full_pre) #transforms all NaNs to mean in each column



# y = full.pop('unix_difference')
# full['new_y'] = 0
# for i, row in full.iterrows():
#     if row['unix_difference'] > 100000:
#         full['new_y'][i] = 1
y = np.array([6.95, 10.45, 18.88, 18.97, 1.13, 5.58, 18.56, 5.16, 7.94, 6.97, 4.05, 10.28, 3.90, 7.07, 7.13, 9.73, 4.80, 5.01, 8.17, 18.57, 9.21, 13.63, 19.32, 15.98, 6.32, 4.61, 0.70, 17.94, 8.56, 16.24, 2.97, 20.88, 19.56, 8.00, 7.42, 5.29, 14.98, 15.60, 5.88, 9.87, 2.25, 10.72, 9.31, 10.97, 6.15, 8.53, 9.01, 4.47, 13.67, 6.95, 15.03, 42.05, 30.82, 6.88, 25.45, 9.15, 17.79, 1.20, 2.97, 8.26, 13.68, 7.33, 0.46, 2.63, 4.10, 10.42, 1.50]).T

'''control'''
rs = np.random.RandomState(2)
d = pd.DataFrame(data=rs.normal(size=(67, 143)))
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(d, y, test_size=0.3, random_state=23)
'''control'''

df_num_X2 = full
# df_num_X2 = df_num_X2.values
X_train, X_test, y_train, y_test = train_test_split(df_num_X2, y, test_size=0.3, random_state=23)


rf = RandomForestRegressor(n_estimators=1000,oob_score=True, max_features=70, n_jobs=3, verbose=False, min_samples_leaf=5, max_depth=7)
rf.fit(X_train, y_train)
'''
# rf2 = RandomForestRegressor(n_estimators=10,oob_score=True, max_features=70, n_jobs=3, verbose=False, min_samples_leaf=5, max_depth=7)



ada = AdaBoostRegressor(n_estimators=1000)
ada.fit(X_train, y_train)

bag = BaggingRegressor(n_estimators=1000, max_features=70, max_samples=20)
bag.fit(X_train, y_train)



'''

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


rf_c = RandomForestRegressor(n_estimators=1000,oob_score=True, max_features=70, n_jobs=3, verbose=False, min_samples_leaf=5, max_depth=7)
rf_c.fit(X_train_c, y_train_c)
y_pred_c = rf_c.predict(X_test_c)
# y_pred_ada = ada.predict(X_test)

# y_pred_bag = bag.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_c = mean_squared_error(y_test_c, y_pred_c)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# mse_ada = mean_squared_error(y_test, y_pred_ada)

# mse_bag = mean_squared_error(y_test, y_pred_bag)



# print 'MSE rf: ', mse_rf
print 'RMSE RF...', np.sqrt(mse_rf)
print 'Control RMSE...', np.sqrt(mse_c)
print 'RMSE LR...', np.sqrt(mse_lr)

'''
print 'OOB_error: ', rf.oob_score_
print "*^"*20
print 'RMSE ada: ', np.sqrt(mse_ada)
print "*^"*20
print 'RMSE bag: ', np.sqrt(mse_bag)




# print y_pred
# df["rf_pred"] = rf.predict(df_rf)
'''
