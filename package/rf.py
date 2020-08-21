import statistics

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RF:
    rf = None
    sc = None
    X_train = None
    y_train = None
    y_std_train = None

    def __init__(self):
        pass

    def train(self, X_train, y_train, std=None):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        y_train_temp = self.y_train.to_numpy(dtype=float)
        if std is None:
            self.y_std_train = statistics.stdev(y_train_temp)
        else:
            self.y_std_train = std
        self.rf = RandomForestRegressor(n_estimators=145, max_depth=30, min_samples_leaf=1).fit(self.X_train,
                                                                                                self.y_train)

    def train_synth(self, X_train, y_train, std=None):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        y_train_temp = self.y_train
        if std is None:
            self.y_std_train = statistics.stdev(y_train_temp)
        else:
            self.y_std_train = std
        self.rf = RandomForestRegressor(n_estimators=145, max_depth=30, min_samples_leaf=1).fit(self.X_train,
                                                                                                self.y_train)
    
    # Note: Not using this predict function for RF

    # This is used in test2() - for making domain in/out predictions. Difference b/w this and getrfmetrics is that
    # it returns predicted value and error while getrfmetrics returns residuals and error
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
        return self.rf.predict(x_pred), error / self.y_std_train

    # alternate predict function that doesn't divide errors by standard deviations
    def predict_no_divide(self, x_test, retstd=True):
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
        return self.rf.predict(x_pred), error

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
        return y_residual / self.y_std_train, error / self.y_std_train
        # Using the stdDev from the paper.
        # return y_residual / 0.4738, error / 0.4738
