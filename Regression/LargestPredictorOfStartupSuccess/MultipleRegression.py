#Data Preprocessing

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

#Importing Data
Data = pd.read_csv("50_Startups.csv")
X = Data.iloc[:, :-1].values
y = Data.iloc[:, 4].values
y = y.reshape(-1, 1)

#Process the Data
Imp = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
Imp.fit(X[:, 0:3])
X[:,0:3] = Imp.transform(X[:,0:3])
LabelEnc_X = LabelEncoder()
X[:, 3] = LabelEnc_X.fit_transform(X[:, 3])
OneHotty = OneHotEncoder(categorical_features = [3])
X = OneHotty.fit_transform(X).toarray()

#Avoiding dummy variable trap
X = X[:, 1:]

#Feature scaling
#Scaler_X = StandardScaler()
#X = Scaler_X.fit_transform(X)
#Scaler_y = StandardScaler()
#y = Scaler_y.fit_transform(y)

#Splitting the data into test and training sets
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Creating naive regressor
Regressor = LinearRegression()
Regressor.fit(X_Train, y_Train)

#y_Prediction = Regressor.predict(X_Test)
#print Scaler_y.inverse_transform(y_Prediction)
#print Scaler_y.inverse_transform(y_Test)
p_value = 0.05
#Backward Elimination
X = np.append(values = X, arr = np.ones((50, 1)).astype(int), axis = 1)
print X
Xopt = X[:, [0, 1, 2, 3, 4, 5]]
Regressor_OLS = sm.OLS(endog = y, exog = Xopt).fit()
print Regressor_OLS.summary()
#x2 has the highest pvalue
Xopt = X[:, [0, 1, 3, 4, 5]]
Regressor_OLS = sm.OLS(endog = y, exog = Xopt).fit()
print Regressor_OLS.summary()
#x1 has the highest pvalue
Xopt = X[:, [0, 3, 4, 5]]
Regressor_OLS = sm.OLS(endog = y, exog = Xopt).fit()
print Regressor_OLS.summary()
#x2 has the highest value
Xopt = X[:, [0, 3, 5]]
Regressor_OLS = sm.OLS(endog = y, exog = Xopt).fit()
print Regressor_OLS.summary()
#x2 has the highest value
Xopt = X[:, [0, 3]]
Regressor_OLS = sm.OLS(endog = y, exog = Xopt).fit()
print Regressor_OLS.summary()
#x3 has the highest value
