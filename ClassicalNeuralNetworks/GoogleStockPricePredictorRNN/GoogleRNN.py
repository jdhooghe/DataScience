import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Importing the data
TrainingData = pd.read_csv("Google_Stock_Price_Train.csv")
TrainingSet = TrainingData.iloc[:, 1:2].values

#Feature scaling
SC = MinMaxScaler()
ScaledTrainingSet = SC.fit_transform(TrainingSet)

#Creating the data structure for the RNN
NumberOfTimeSteps = 60
X_Train = []
y_Train = []

DropoutRate = 0.2

for i in range(60, ScaledTrainingSet.size):
    X_Train.append(ScaledTrainingSet[i-60:i, 0])
    y_Train.append(ScaledTrainingSet[i, 0])
    
X_Train, y_Train = np.array(X_Train), np.array(y_Train)

Shape = (X_Train.shape[0], X_Train.shape[1], 1)

X_Train = np.reshape(X_Train, Shape)

#Creating the RNN
def CreateRegressor():
    Regressor = Sequential()
    Regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_Train.shape[1], 1)))
    Regressor.add(Dropout(DropoutRate))
    Regressor.add(LSTM(units=50, return_sequences=True))
    Regressor.add(Dropout(DropoutRate))
    Regressor.add(LSTM(units=50, return_sequences=True))
    Regressor.add(Dropout(DropoutRate))
    Regressor.add(LSTM(units=50))
    Regressor.add(Dropout(DropoutRate))
    Regressor.add(Dense(units=1))
    
    return Regressor

#Initialization of the RNN
Reg = CreateRegressor()
Reg.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training
Reg.fit(X_Train, y_Train, epochs = 100, batch_size = 32)

#Getting the test data
TestData = pd.read_csv("Google_Stock_Price_Test.csv")
TestSet = TestData.iloc[:, 1:2].values

#Getting the predicted stock prices of Jan 2017. Do not change the test value scalings and do not change the test scaling.
AdditionOfDataSets = pd.concat((TrainingData['Open'], TestData['Open']), axis=0) #Vertical axis
CutData_ = AdditionOfDataSets.values
CutData = CutData_.reshape(-1, 1)
CutData = SC.transform(CutData)
X_Test = []

for i in range(len(TrainingData), CutData.size):
    X_Test.append(CutData[i-60:i, 0])
X_Test = np.array(X_Test)
Shape = (X_Test.shape[0], X_Test.shape[1], 1)
X_Test = np.reshape(X_Test, Shape)

Prediction = Reg.predict(X_Test)
Prediction = SC.inverse_transform(Prediction)

plt.plot(TestSet, color='r', label="real google stock price")
plt.plot(Prediction, color='b', label="predicted google stock price")
plt.title("Google Stock Price")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()