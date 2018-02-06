#Data Preprocessing

#Importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Importing Data
Data = pd.read_csv("Position_Salaries.csv")
X = Data.iloc[:, 1:2].values
y = Data.iloc[:, 2].values

Regressor = RandomForestRegressor(n_estimators = 10000, random_state = 0)
Regressor.fit(X, y)

print Regressor.predict(6.5)
