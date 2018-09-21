# Data Preprocessing

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.cross_validation as sklcv

# Read in data
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Fix missing data issues
imputer = sklp.Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Deal with categorical data
lbEncIv = sklp.LabelEncoder()
X[:,0] = lbEncIv.fit_transform(X[:,0])
oneHotEncIv = sklp.OneHotEncoder(categorical_features = [0])
X = oneHotEncIv.fit_transform(X).toarray()

lbEncDv = sklp.LabelEncoder()
y = lbEncIv.fit_transform(y)

# Split dataset into training and test datasets ot validate ML model
XTrain,XTest,yTrain,yTest = sklcv.train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
scX = sklp.StandardScaler()
XTrain = scX.fit_transform(XTrain)
XTest = scX.transform(XTest)

print(XTrain)
print(XTest)
