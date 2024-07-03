
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

"""To execute the code in the above cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut "Command/Ctrl+Enter". To edit the code, just click the cell and start editing.

Variables that you define in one cell can later be used in other cells:
"""

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

"""You can import your own data into Colab notebooks from your Google Drive account, including from spreadsheets, as well as from Github and many other sources. To learn more about importing data, and how Colab can be used for data science, see the links below under [Working with Data](#working-with-data)."""

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
