import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Getting the dataset
dataset = pd.read_csv('_haijinlogfeaturesnobarrier_alldata.csv', error_bad_lines=False, sep=',', header=None)
dataset.head()
dataset = dataset.drop([0])

# Dividing dataset
X = dataset.iloc[:, 0:24]
y = dataset.iloc[:, 27]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# model training
regressor = RandomForestRegressor(n_estimators=15, random_state=8)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print(y_test)

y_test = y_test.to_numpy(dtype=float)
y_residual = abs(y_pred - y_test)

# Evaluating algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Prediction standard deviation
error = []
for x in range(len(X_test)):
    preds = []
    for pred in regressor.estimators_:
        preds.append(pred.predict([X_test[x]])[0])
    error.append(statistics.stdev(preds))
error = np.array(error)


# glennfun(y_residual,error,statistics.stdev(y_test))


