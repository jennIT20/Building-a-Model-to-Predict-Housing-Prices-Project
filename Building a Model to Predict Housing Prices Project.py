
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

data = pd.read_csv('houseSmallData.csv')
data.shape
train = data.iloc[0:20,:]
train.head()

# investigate Sale Price
train['SalePrice']


plt.hist(train['SalePrice'])


# select numeric columns
numeric = train.select_dtypes(include=[np.number])
numeric.shape

# @title Default title text
corr = numeric.corr()
cols = corr['SalePrice'].sort_values(ascending=False)[1:6].index
cols

X = train[cols]
Y = train ['SalePrice']
X = X.drop(['SalePrice'], axis = 1)
X


# build Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)
predictions = model.predict(X)
print('predictions', predictions)

# How good is the model
model.score(X, Y)

#scatter plot of predictions
plt.scatter(Y, predictions)
