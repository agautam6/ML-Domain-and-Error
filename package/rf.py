from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statistics
import numpy as np


class RF:
    rf = None
    sc = None
    X_train = None
    y_train = None
    y_std_train = None

    def __init__(self):
        pass

    def train(self, X_train, y_train):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        y_train_temp = self.y_train.to_numpy(dtype=float)
        self.y_std_train = statistics.stdev(y_train_temp)
        self.rf = RandomForestRegressor(n_estimators=15, random_state=8).fit(self.X_train, self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        if retstd is False:
            return self.rf.predict(x_pred)
        error = []
        preds = []
        for x in range(len(x_pred)):
            preds = []
            for pred in self.rf.estimators_:
                preds.append(pred.predict([x_pred[x]])[0])
            error.append(statistics.stdev(preds))
        error = np.array(error)
        return self.rf.predict(x_pred), error/self.y_std_train

    def getrfmetrics(self, X_test, y_test):
        X_pred = self.sc.transform(X_test)
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
