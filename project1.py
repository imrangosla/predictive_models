import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Dataset.csv'    )
feature_cols = ['Tm','Pr','Th','Sv']
X = data[feature_cols]
y = data.Idx


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.4)

rmse_array = []
degrees = np.arange(1, 10)
min_rmse, minDegree = 1e10, 0

for deg in degrees:
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_train)

    # instantiate linear regression model
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)

    # predict and compare with the test data
    X_poly_test = poly_features.fit_transform(X_test)
    polyPrediction = poly_reg.predict(X_poly_test)

    poly_mse = mean_squared_error(y_test, polyPrediction)
    poly_rmse = np.sqrt(poly_mse)
    rmse_array.append(poly_rmse)

    # cross validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        minDegree = deg

print('FOR LINEAR MODEL: \nOptimal Order of Polynomial: {}\nRMSE: {}'.format(minDegree, min_rmse))
