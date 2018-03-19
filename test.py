import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data= pd.read_csv('Dataset.csv', header='infer', index_col=False, dtype=float)

columns = data.columns.tolist()
print("columns: " )
print(columns)

#print(data.corr()["Idx"])

y = data.pop('Idx')
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print("Mean Squared Model Error:")
print(mean_squared_error(predictions, y_test))

m = model.coef_[0]
b = model.intercept_

print("Equation: ")
print(' y = {0} * x + {1}'.format(m, b))
#pyplot.show()
