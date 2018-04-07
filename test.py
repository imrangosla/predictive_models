# conventional way to import
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics

# read CSV (comma separated value) file and save the results
data = pd.read_csv('Dataset.csv')

# visualize th relationship between the features and the response using scatterplots
# sns.pairplot(data, x_vars=['Tm','Pr','Th','Sv'], y_vars='Idx', size=7, aspect=0.7, kind='reg')

# show the plot
# plt.show()

# use the list to collect a subset of the original DataFrame
feature_cols = ['Tm','Pr','Th','Sv']
X = data[feature_cols]

# select a Series from the DataFrame
y = data.Idx

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)#test_size=0.2

# instantiate model
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients of the line of best fit)
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print('////////////////LINEAR REGRESSION w/ TEST SPLIT////////////////')
print('Intercept:',linreg.intercept_)
print('Coefficients:',linreg.coef_)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

# computing the root mean squared error for our Idx predictions (RMSE is in the units of our response variable) (smaller is better) 
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# remove features to compare model accuracy
feature_cols = ['Pr','Th','Sv']
X = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('Without Tm:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

feature_cols = ['Tm','Th','Sv']
X = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('Without Pr:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

feature_cols = ['Tm','Pr','Sv']
X = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('Without Th:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

feature_cols = ['Tm','Pr','Th']
X = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print('Without Sv:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# 10-fold cross-validation
print('/////////LINEAR REGRESSION w/ 10-FOLD CROSS VALIDATION/////////')
X = data[['Tm','Pr','Th','Sv']]
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

# fix the signs of mean squared errors and convert to ROOT mse
scores = np.sqrt(scores * -1)
print('Cross Validation Scores:',scores)

# use average accuracy as an estimate of out-of-sample accuracy
print('Average Score (RMSE):',scores.mean())

# remove features to compare model accuracy
feature_cols = ['Pr','Th','Sv']
X = data[feature_cols]
print('Without Tm:',np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

feature_cols = ['Tm','Th','Sv']
X = data[feature_cols]
print('Without Pr:',np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

feature_cols = ['Tm','Pr','Sv']
X = data[feature_cols]
print('Without Th:',np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

feature_cols = ['Tm','Pr','Th']
X = data[feature_cols]
print('Without Sv:',np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

# Conclusion
print('///////////////////////MODEL COMPARISON//////////////////////')
print(' Linear Regression model without removing any of the features \n results in a more accurate model because it minimizes the root mean squared error \n (using cross validation reduces the variance associated with a single trial of train/test split)')

feature_cols = ['Tm','Pr','Th','Sv']
X = data[feature_cols]
y = data.Idx
clf = linear_model.LassoLars(alpha=.1)
clf.fit(X,y)
print('////////////////LARS LASSO////////////////')
print('Intercept:',clf.intercept_)
print('Coefficients:',clf.coef_)
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
clf.fit(X,y)
print('////////////////RIDGE REGRESSION////////////////')
print('Intercept:',clf.intercept_)
print('Coefficients:',clf.coef_)