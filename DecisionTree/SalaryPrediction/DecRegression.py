#Data Preprocessing

#Importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Importing Data
Data = pd.read_csv("Position_Salaries.csv")
X = Data.iloc[:, 1:2].values
y = Data.iloc[:, 2].values

#Decision Decision Tree Regressor
Regressor = DecisionTreeRegressor(random_state = 0)
Regressor.fit(X, y)

Y_Prediction = Regressor.predict(6.5)
print Y_Prediction
