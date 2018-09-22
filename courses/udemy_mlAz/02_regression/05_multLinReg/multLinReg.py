# Multiple Linear Regression

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm
import statsmodels.formula.api as sm

# Read in data
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Deal with categorical data
lbEncIv = sklp.LabelEncoder()
X[:,3] = lbEncIv.fit_transform(X[:,3])
oneHotEncIv = sklp.OneHotEncoder(categorical_features = [3])
X = oneHotEncIv.fit_transform(X).toarray()

#Avoid over-fitting by removing one dummy variable
X = X[:,1:]

# Split dataset into training and test datasets ot validate ML model
XTrain,XTest,yTrain,yTest = sklcv.train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression
regressor = skllm.LinearRegression()
regressor.fit(XTrain,yTrain)

# Predicting test set results
yPred = regressor.predict(XTest)

# Building optimal model using Backward Elimination
X = np.append(arr = np.ones((len(X),1)).astype(int) , \
                values = X, axis = 1)
optX = X
optVarIdx = range(len(X[0]))
sigLvl = 0.05
while len(optX[0]) > 0:
    regressorOls = sm.OLS(endog = y, exog = optX).fit()
    pVals = regressorOls.pvalues
    maxP = max(pVals)
    if maxP > sigLvl:
        rmIdx = pVals.argmax()
        optX = np.delete(optX,rmIdx,1)
        optVarIdx = np.delete(optVarIdx,rmIdx)
        print(optVarIdx)
    else:
        break

# Set X to ne optimal X value
X = optX

# Split dataset into training and test datasets ot validate ML model
XTrain,XTest,yTrain,yTest = sklcv.train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression with optimal X
regressor = skllm.LinearRegression()
regressor.fit(XTrain,yTrain)

# Predicting test set results
yPred = regressor.predict(XTest)
