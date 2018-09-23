# Logistic Regression

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr
import pandas as pd
import sklearn.preprocessing as sklp
import sklearn.model_selection as sklcv
import sklearn.linear_model as skllm
import sklearn.metrics as sklmet

# Read in data
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Split dataset into training and test datasets ot validate ML model
XTrain,XTest,yTrain,yTest = sklcv.train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
scX = sklp.StandardScaler()
XTrain = scX.fit_transform(XTrain)
XTest = scX.transform(XTest)

# Fitting logistic linear regression
classifier = skllm.LogisticRegression(random_state = 0)
classifier.fit(XTrain,yTrain)

# Predicting test set results
yPred = classifier.predict(XTest)
print(yPred)

# Creating confusion matrix
cm = sklmet.confusion_matrix(yTest, yPred)
print(cm)

# Visualizing the Training set results
XSet, ySet = XTrain, yTrain
X1, X2 = np.meshgrid(
         np.arange(start = XSet[:,0].min() - 1, stop = XSet[:,0].max() + 1, step = 0.01), \
         np.arange(start = XSet[:,1].min() - 1, stop = XSet[:,1].max() + 1, step = 0.01) \
)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = pltclr.ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(XSet[ySet == j, 0], XSet[ySet == j, 1],
                c = pltclr.ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression - Ad Campaign (Training set)')
plt.xlabel('Age [years]')
plt.ylabel('Estimated Salary [USD]')
plt.legend()
plt.show()
