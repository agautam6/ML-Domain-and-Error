import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from tabulate import tabulate

from package import gpr
from package import rf
from package import io
from package import testhelper as th


# Test Description: training GPR and RF models from 'alldata' (70% train 30% test) and plotting
# residual/y_std vs error/y_std for test data for both models
def test1():
    data = io.importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = io.getdata(data)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
    GPR = gpr.GPR()
    GPR.train(X_train, y_train)
    res, sigma = GPR.getgprmetrics(X_test, y_test)
    th.plot(res, sigma, "GPR", 8)
    RF = rf.RF()
    RF.train(X_train, y_train)
    res, sigma = RF.getrfmetrics(X_test, y_test)
    th.plot(res, sigma, "RF", 8)


# Test Description: training GPR and RF models from 'alldata' or 'alldata_no_Pd', making predictions for 'Pd_only' and
# tabulating domain IN/OUT
def test2():
    data = io.importdata('_haijinlogfeaturesnobarrier_alldata_no_Pd.csv')
    data = io.getdata(data)
    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train)
    test_data = io.importdata('_haijinlogfeatures_Pd_only.csv')
    test_data = test_data.drop([0])
    final_list = [i + "-" + j for i, j in zip(test_data[24].values, test_data[25].values)]
    test_data = test_data.drop([24, 25], axis=1)
    # Define GPR and RF errors
    pred, GPR_errors = GPR.predict(test_data, True)
    RF = rf.RF()
    RF.train(X_train, y_train)
    pred, RF_errors = RF.predict(test_data, True)
    print(final_list)
    predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, 37)]
    results = [(final_list[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in range(0, 37)]
    print(tabulate(results, headers=["Material", "In domain?", "GPR predicted error", "RF predicted error"]))


# Test for getting plots using cross validation

def test3(k, n):
    data = io.importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = io.getdata(data)
    X_CV = data.iloc[:, :-1]
    Y_CV = data.iloc[:, -1]

    # rf_res = np.array((0, 100))
    rf_res = np.asarray([])
    rf_sigma = np.asarray([])
    gpr_res = np.asarray([])
    gpr_sigma = np.asarray([])

    # This will do repeated cross validation with the given k splits and n repeats
    rkf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=2652124)
    for train_index, test_index in rkf.split(data):
        X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
        y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
        GPR = gpr.GPR()
        GPR.train(X_train, y_train)
        res, sigma = GPR.getgprmetrics(X_test, y_test)
        gpr_res = np.concatenate((gpr_res, res), axis=None)
        gpr_sigma = np.concatenate((gpr_sigma, sigma), axis=None)
        RF = rf.RF()
        RF.train(X_train, y_train)
        res, sigma = RF.getrfmetrics(X_test, y_test)
        rf_res = np.concatenate((rf_res, res), axis=None)
        rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)

    th.plot(gpr_res, gpr_sigma, "GPR", 20)
    th.plot(rf_res, rf_sigma, "RF", 20)


def main():
    test1()
    test2()
    test3(5, 20)


if __name__ == "__main__":
    main()
