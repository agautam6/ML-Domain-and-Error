import numpy as np
import statistics
from sklearn.model_selection import RepeatedKFold
from package import io
from package import rf
from package import testhelper as th


def test(k, n):
    data = io.importdata('data/PVstability_Weipaper_alldata_featureselected.csv')
    data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])
    X_CV = data.iloc[:, 1:]
    Y_CV = data.iloc[:, 0]
    y_std = statistics.stdev(Y_CV.to_numpy(dtype=float))
    rf_res = np.asarray([])
    rf_sigma = np.asarray([])
    rkf = RepeatedKFold(n_splits=k, n_repeats=n)
    print("Folds: {} Iterations: {}".format(k, n))
    ctr = 0
    for train_index, test_index in rkf.split(data):
        print(ctr)
        ctr += 1
        X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
        y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
        RF = rf.RF()
        RF.train(X_train, y_train, std=y_std)
        res, sigma = RF.getrfmetrics(X_test, y_test)
        rf_res = np.concatenate((rf_res, res), axis=None)
        rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)
    th.RF_plot(rf_res, rf_sigma, "RF", 20, filename="RF_{}-fold_{}-iter".format(k, n))


if __name__ == "__main__":
    for it in [1, 2, 5, 10]:
        for fold in [2, 3, 5, 10, 20, 50]:
            test(fold, it)
