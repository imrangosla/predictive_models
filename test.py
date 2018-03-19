import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data= pd.read_csv('Dataset.csv', header='infer', index_col=False, dtype=float)

columns = data.columns.tolist()
print("columns: " )
print(columns)

#print(data.corr()["Idx"])

y = data.pop('Idx')
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#model = LinearRegression()

model.fit()
print(data.shape)
print(y.shape)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
#pyplot.show()
