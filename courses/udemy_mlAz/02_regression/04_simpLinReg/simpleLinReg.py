# Simple Linear Regression
# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm

# Read in data
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Split dataset into training and test datasets ot validate ML model
XTrain,XTest,yTrain,yTest = sklcv.train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting simple linear regression
regressor = skllm.LinearRegression()
regressor.fit(XTrain,yTrain)

# Predicting test set results
yPred = regressor.predict(XTest)

# Plot results for training set
trainPts = plt.scatter(XTrain,yTrain, color = 'red')
testPts = plt.scatter(XTest,yTest, color = 'green')
regLine, = plt.plot(XTrain,regressor.predict(XTrain), color = 'blue')
plt.title('Salary vs. Experience')
plt.xlabel('Experience [years]')
plt.ylabel('Salary [USD]')
plt.legend([trainPts,testPts,regLine],['Training Set','Test Set','Regression Line'])
plt.show()
