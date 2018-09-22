# Polynomial Regression

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm
import statsmodels.formula.api as sm

# Read in data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting simple linear regression
regressorLin = skllm.LinearRegression()
regressorLin.fit(X,y)

# Transforming X to polynomial IVs
polyReg = sklp.PolynomialFeatures(degree = 4)
polyX = polyReg.fit_transform(X)

# Fitting polynomial regression
regressorPoly = skllm.LinearRegression()
regressorPoly.fit(polyX,y)

# Predicting lin reg results for 6.5
yPredLin = regressorLin.predict(6.5)
print(yPredLin)

# Predicting poly reg results for 6.5
yPredLin = regressorPoly.predict(polyReg.fit_transform(6.5))
print(yPredLin)

# Plot results for linear regression
obsSalaries = plt.scatter(X,y, color = 'red')
regLine, = plt.plot(X,regressorLin.predict(X), color = 'blue')
plt.title('Salary vs. Level  (Linear Regression)')
plt.xlabel('Level [none]')
plt.ylabel('Salary [USD]')
plt.legend([obsSalaries,regLine],['Actual Salaries','Regression Line'])
plt.show()

# Plot results for polynomial regression
gridX = np.arange(min(X), max(X), 0.1)
gridX = gridX.reshape((len(gridX),1))
obsSalaries = plt.scatter(X,y, color = 'red')
regLine, = plt.plot(gridX,regressorPoly.predict(polyReg.fit_transform(gridX)), color = 'blue')
plt.title('Salary vs. Level (Polynomial Regression)')
plt.xlabel('Level [none]')
plt.ylabel('Salary [USD]')
plt.legend([obsSalaries,regLine],['Actual Salaries','Regression Line'])
plt.show()
