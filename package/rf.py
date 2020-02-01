from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statistics
import numpy as np


class RF:
    rf = None
    sc = None
    X_train = None
    y_train = None

    def __init__(self):
        pass

    def train(self, X_train, y_train):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        self.rf = RandomForestRegressor(n_estimators=15, random_state=8).fit(self.X_train, self.y_train)

    def predict(self, x_test):
        x_pred = self.sc.fit_transform(x_test)
        return self.rf.predict(x_pred)

    def getrfmetrics(self, X_test, y_test):
        X_pred = self.sc.fit_transform(X_test)
        y_pred = self.rf.predict(X_pred)
        y_test1 = y_test.to_numpy(dtype=float)
        y_residual = abs(y_pred - y_test1)
        error = []
        for x in range(len(X_pred)):
            preds = []
            for pred in self.rf.estimators_:
                preds.append(pred.predict([X_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        y_std = statistics.stdev(y_test1)
        return y_residual/y_std, error/y_std
