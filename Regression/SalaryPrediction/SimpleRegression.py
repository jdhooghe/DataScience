#Data Preprocessing

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#Importing Data
Data = pd.read_csv("Salary_Data.csv")
X = Data.iloc[:, 0].values
y = Data.iloc[:, 1].values

#Splitting the data into test and training sets
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Creating the regressor
X_Train = X_Train.reshape(-1, 1)
y_Train = y_Train.reshape(-1, 1)
X_Test = X_Test.reshape(-1, 1)
y_Test = y_Test.reshape(-1, 1)
Regressor = LinearRegression()
Regressor.fit(X_Train, y_Train)

#Prediction of y_Test
y_Test_Prediction = Regressor.predict(X_Test)
plt.scatter(X_Test, y_Test, color = 'red')
plt.scatter(X_Test, y_Test_Prediction, color = 'blue')
plt.title("Prediction vs. Actual of Salary Expectations")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
