import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from sklearn.svm import SVR

data = pd.read_csv('Dataset.csv'    )
feature_cols = ['Tm','Pr','Th','Sv']
X = data[feature_cols]
y = data.Idx


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rmse_array = []
degrees = np.arange(1, 10)
min_rmse, minDegree = 1e10, 0


for deg in degrees:
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_train)

    # instantiate linear regression model
    poly_reg = LinearRegression(fit_intercept=True)
    #poly_reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
    poly_reg.fit(X_poly_train, y_train)

    # predict and compare with the test data
    X_poly_test = poly_features.fit_transform(X_test)
    polyPrediction = poly_reg.predict(X_poly_test)

    # calculate the mean squared error
    poly_mse = mean_squared_error(y_test, polyPrediction)
    poly_rmse = np.sqrt(poly_mse)
    rmse_array.append(poly_rmse)

    # cross validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        minDegree = deg
        minIntercept = poly_reg.intercept_
        minCoefficients = poly_reg.coef_

print(polyPrediction.shape)

plt.scatter(X_poly_test[:,3], polyPrediction, color='k')
plt.scatter(data['Sv'], data['Idx'], color='g')
plt.show()

print('FOR LINEAR MODEL: \nOptimal Order of Polynomial: {}\nRMSE: {}'.format(minDegree, min_rmse))
print('Intercept:', minIntercept)
print('Coefficients:', minCoefficients)
