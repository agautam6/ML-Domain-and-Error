import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from tabulate import tabulate

from package import gpr
from package import rf


def plot(res, sigma, model_name, number_of_bins):
    # Define input data -- divide by standard deviation
    # model_errors = sigma / stdev
    # abs_res = res / stdev
    # The above code for scaling with stdev has been moved to GPR.getgprmetrics and RF.getrfmetrics
    abs_res = res
    model_errors = sigma

    # Create initial scatter plot
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("%s Absolute residuals / dataset stdev" % (model_name))
    plt.title("%s Absolute Residuals vs. Model Errors" % (model_name))
    plt.plot(model_errors, abs_res, '.', color='blue');

    plt.show()

    # Histogram of RF error bin counts
    plt.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("Counts")
    plt.title("%s Bin Counts" % (model_name));

    plt.show()

    # Set bins for calculating RMS
    upperbound = np.amax(model_errors)
    lowerbound = np.amin(model_errors)
    bins = np.linspace(lowerbound, upperbound, number_of_bins, endpoint=False)

    # Create a vector determining bin of each data point
    digitized = np.digitize(model_errors, bins)

    # Record which bins contain data (to avoid trying to do calculations on empty bins)
    bins_present = []
    for i in range(1, number_of_bins + 1):
        if i in digitized:
            bins_present.append(i)

    # Calculate RMS of the absolute residuals
    RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]

    # Set the x-values to the midpoint of each bin
    bin_width = bins[1] - bins[0]
    binned_model_errors = np.zeros(len(bins_present))
    for i in range(0, len(bins_present)):
        curr_bin = bins_present[i]
        binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

    # Fit a line to the data
    model = LinearRegression(fit_intercept=False)
    model.fit(binned_model_errors[:, np.newaxis],
              RMS_abs_res)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    xfit = binned_model_errors
    yfit = model.predict(xfit[:, np.newaxis])

    # Calculate r^2 value
    r_squared = r2_score(RMS_abs_res, yfit)
    # Calculate slope
    slope = model.coef_

    # Create RMS scatter plot
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("%s RMS Absolute residuals / dataset stdev" % (model_name))
    plt.ylim(0, 1)
    plt.title("%s RMS Absolute Residuals vs. Model Errors" % (model_name))
    plt.text(0.2, 0.8, 'r^2 = %f' % (r_squared))
    plt.text(0.2, 0.7, 'slope = %f' % (slope))
    plt.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
    plt.plot(xfit, yfit);

    plt.show()


# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data


# Sanitize data. Temporarily dropping columns E_regression, Material Composition
def getdata(data):
    data = data.drop([24, 25, 26], axis=1)
    data = data.drop([0])
    return data


# Fetch data with filename
def importdatanames(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data


def predictdomain(GPR_error, RF_error):
    if GPR_error < 0.8 and RF_error < 0.8:
        return 1
    else:
        return 0


# Test Description: training GPR and RF models from 'alldata' (70% train 30% test) and plotting
# residual/y_std vs error/y_std for test data for both models
def test1():
    data = importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = getdata(data)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
    GPR = gpr.GPR()
    GPR.train(X_train, y_train)
    res, sigma = GPR.getgprmetrics(X_test, y_test)
    plot(res, sigma, "GPR", 8)
    RF = rf.RF()
    RF.train(X_train, y_train)
    res, sigma = RF.getrfmetrics(X_test, y_test)
    plot(res, sigma, "RF", 8)


# Test Description: training GPR and RF models from 'alldata' or 'alldata_no_Pd', making predictions for 'Pd_only' and
# tabulating domain IN/OUT
def test2():
    data = importdata('_haijinlogfeaturesnobarrier_alldata_no_Pd.csv')
    data = getdata(data)
    # X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0)
    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train)
    test_data = importdata('_haijinlogfeatures_Pd_only.csv')
    test_data = test_data.drop([24, 25], axis=1)
    test_data = test_data.drop([0])
    # Define GPR and RF errors
    pred, GPR_errors = GPR.predict(test_data, True)
    RF = rf.RF()
    RF.train(X_train, y_train)
    pred, RF_errors = RF.predict(test_data, True)
    # GPR_errors = np.random.random_sample(37, )
    # RF_errors = np.random.random_sample(37, )
    data = importdatanames('_haijinlogfeatures_Pd_only.csv')
    list = []
    for element in data[24]:
        list.append(element)
    del list[0]
    list2 = []
    for element in data[25]:
        list2.append(element)
    del list2[0]
    final_list = []
    for i in range(0, 37):
        first = list[i]
        second = list2[i]
        combined = first + "-" + second
        final_list.append(combined)
    print(final_list)
    predictions = [predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, 37)]
    results = [(final_list[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in range(0, 37)]
    print(tabulate(results, headers=["Material", "In domain?", "GPR predicted error", "RF predicted error"]))


def test3():
    data = importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = getdata(data)
    X_CV = data.iloc[:, :-1]
    Y_CV = data.iloc[:, -1]

    # rf_res = np.array((0, 100))
    rf_res = np.asarray([])
    rf_sigma = np.asarray([])
    gpr_res = np.asarray([])
    gpr_sigma = np.asarray([])
    counter = 1

    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=2652124)
    for train_index, test_index in rkf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        print(counter)
        counter = counter + 1
        X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
        y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
        GPR = gpr.GPR()
        GPR.train(X_train, y_train)
        res, sigma = GPR.getgprmetrics(X_test, y_test)
        # np.append(gpr_res, res)
        # np.append(gpr_sigma, sigma)
        gpr_res = np.concatenate((gpr_res, res), axis=None)
        gpr_sigma = np.concatenate((gpr_sigma, sigma), axis=None)
        # plot(res, sigma, "GPR", 8)
        RF = rf.RF()
        RF.train(X_train, y_train)
        res, sigma = RF.getrfmetrics(X_test, y_test)
        rf_res = np.concatenate((rf_res, res), axis=None)
        rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)
        # plot(res, sigma, "RF", 8)

    plot(gpr_res, gpr_sigma, "GPR", 20)
    plot(rf_res, rf_sigma, "RF", 20)


def main():
    test1()
    test2()
    test3()


if __name__ == "__main__":
    main()
