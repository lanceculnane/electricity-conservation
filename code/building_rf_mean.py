import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer


full_pre = pd.read_csv("buildingfull.csv")
# full_pre.fillna(0, inplace=True)
full = full_pre.ix[:66,2:]
# Create our imputer to replace missing values with the mean e.g.

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)



imp2 = imp.fit(full_pre.ix[:66,2:]) #fits it so it knows mean in each column
filled = imp.transform(full_pre.ix[:66,2:]) #transforms all NaNs to mean in each column


# y = full.pop('unix_difference')
# full['new_y'] = 0
# for i, row in full.iterrows():
#     if row['unix_difference'] > 100000:
#         full['new_y'][i] = 1
# print full.shape
# print full
y = np.array([6.95, 10.45, 18.88, 18.97, 1.13, 5.58, 18.56, 5.16, 7.94, 6.97, 4.05, 10.28, 3.90, 7.07, 7.13, 9.73, 4.80, 5.01, 8.17, 18.57, 9.21, 13.63, 19.32, 15.98, 6.32, 4.61, 0.70, 17.94, 8.56, 16.24, 2.97, 20.88, 19.56, 8.00, 7.42, 5.29, 14.98, 15.60, 5.88, 9.87, 2.25, 10.72, 9.31, 10.97, 6.15, 8.53, 9.01, 4.47, 13.67, 6.95, 15.03, 42.05, 30.82, 6.88, 25.45, 9.15, 17.79, 1.20, 2.97, 8.26, 13.68, 7.33, 0.46, 2.63, 4.10, 10.42, 1.50]).T



df_num_X2 = filled
# df_num_X2 = df_num_X2.values
X_train, X_test, y_train, y_test = train_test_split(df_num_X2, y, test_size=0.30, random_state=23)


rf = RandomForestRegressor(n_estimators=1000,oob_score=True, max_features=5, n_jobs=3, verbose=True, min_samples_leaf=7, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print 'MSE: ', mse
print 'RMSE...', np.sqrt(mse)
print 'OOB_error: ', rf.oob_score_
print y_pred
# df["rf_pred"] = rf.predict(df_rf)
# print full.head()
