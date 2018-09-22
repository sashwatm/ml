# SVR

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm
import statsmodels.formula.api as sm
import sklearn.svm as sksvm

# Read in data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values

# Feature Scaling
scX = sklp.StandardScaler()
scy = sklp.StandardScaler()
X = scX.fit_transform(X)
y = scy.fit_transform(y)

# Fitting SVR
regressorSvr = sksvm.SVR(kernel = 'rbf')
regressorSvr.fit(X,y)

# Predicting poly reg results for 6.5
yPred = scy.inverse_transform(regressorSvr.predict(scX.transform(np.array([[6.5]]))))
print(yPred)

# Plot results for Support Vector Regression
obsSalaries = plt.scatter(X,scy.inverse_transform(y), color = 'red')
regLine, = plt.plot(X,scy.inverse_transform(regressorSvr.predict(X)), color = 'blue')
plt.title('Salary vs. Level (Support Vector Regression)')
plt.xlabel('Level [none]')
plt.ylabel('Salary [USD]')
plt.legend([obsSalaries,regLine],['Actual Salaries','Regression Line'])
plt.show()
