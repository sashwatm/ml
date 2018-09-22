# Decision Tree Regression

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm
import statsmodels.formula.api as sm
import sklearn.tree as sktree

# Read in data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting Decision Tree Regression
regressor = sktree.DecisionTreeRegressor(criterion = 'mse', random_state = 0)
regressor.fit(X,y)

# Predicting results for 6.5
yPred = regressor.predict(6.5)
print(yPred)

# Plot results for Decision Tree Regression
gridX = np.arange(min(X), max(X), 0.01)
gridX = gridX.reshape((len(gridX),1))
obsSalaries = plt.scatter(X,y, color = 'red')
regCurve, = plt.plot(gridX,regressor.predict(gridX), color = 'blue')
plt.title('Salary vs. Level (Binary Tree Regression)')
plt.xlabel('Level [none]')
plt.ylabel('Salary [USD]')
plt.legend([obsSalaries,regCurve],['Actual Salary','Regression Curve'])
plt.show()
