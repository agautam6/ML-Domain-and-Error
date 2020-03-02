import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
import math
from matplotlib.ticker import PercentFormatter


def GPR_plot(res, sigma, model_name, number_of_bins, filename=None):
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

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot1.png".format(filename))
        plt.clf()

    # Histogram of RF error bin counts
    plt.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("Counts")
    plt.title("%s Bin Counts" % (model_name));

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot2.png".format(filename))
        plt.clf()

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

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot3.png".format(filename))
        plt.clf()


def RF_plot(res, sigma, model_name, number_of_bins, filename=None):
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

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot1.png".format(filename))
        plt.clf()

    # Histogram of RF error bin counts
    plt.hist(model_errors, bins=number_of_bins, color='blue', edgecolor='black')
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("Counts")
    plt.title("%s Bin Counts" % (model_name));

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot2.png".format(filename))
        plt.clf()

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

    # Cutoff value for fitting line on RMS graph
    cutoff_value = 1.0

    # Find what bin the cutoff value defined above is in
    cutoff_bin = np.digitize(cutoff_value, bins)

    # Fit a line to the data
    model = LinearRegression(fit_intercept=False)
    model.fit(binned_model_errors[0:cutoff_bin, np.newaxis],
              RMS_abs_res[0:cutoff_bin])  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    xfit = binned_model_errors[0:cutoff_bin]
    yfit = model.predict(xfit[:, np.newaxis])

    # Calculate r^2 value
    r_squared = r2_score(RMS_abs_res[0:cutoff_bin], yfit)
    # Calculate slope
    slope = model.coef_

    # Create RMS scatter plot
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("%s RMS Absolute residuals / dataset stdev" % (model_name))
    plt.ylim(0, 1)
    plt.title("%s RMS Absolute Residuals vs. Model Errors" % (model_name))
    plt.text(0.2, 0.8, 'r^2 = %f' % (r_squared))
    plt.text(0.2, 0.7, 'slope = %f' % (slope))
    plt.plot(binned_model_errors[0:cutoff_bin], RMS_abs_res[0:cutoff_bin], 'o', color='blue')
    plt.plot(binned_model_errors[cutoff_bin:], RMS_abs_res[cutoff_bin:], 'o', color='red')  
    plt.plot(xfit, yfit);

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot3.png".format(filename))
        plt.clf()


def predictdomain(GPR_error, RF_error):
    if GPR_error < 0.8 and RF_error < 0.8:  
        return 1
    else:
        return 0


def getcontribution(GPR_error, RF_error):
    if GPR_error < 0.8 and RF_error < 0.8:
        return 0
    elif RF_error < 0.8 <= GPR_error:
        return 1
    elif GPR_error < 0.8 <= RF_error:
        return 2
    else:
        return 3


# Expects non-empty 'data'
def plotrstatwithgaussian(data, _stacked=True, _label=None, filename=None,
                          _xlabel=None, _ylabel=None, _bincount=10, _title=None):
    # _weights = None
    if not isinstance(data[0], list):  # checking for multiple data sets with only 1st element instead of all()
        # if len(data) is not 0:
        #     _weights = [1 / len(data)] * len(data)
        (mu, sigma) = stats.norm.fit(data)
    else:
        # total = 0
        # for i in range(0, len(data)):
        #     total += len(data[i])
        # if total is not 0:
        #     _weights = w = [[1 / total] * len(data[i]) for i in range(0, len(data))]
        (mu, sigma) = stats.norm.fit([val for sublist in data for val in sublist])
    # n, bins, patches = plt.hist(data, weights=_weights, label=_label, stacked=_stacked)
    n, bins, patches = plt.hist(data, density=True, label=_label, stacked=_stacked, bins=_bincount)
    x = np.linspace(-6, 6, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Gaussian mu: {} std: {}'.format(round(mu, 2), round(sigma, 2)))
    plt.ylabel(_ylabel)
    plt.xlabel(_xlabel)
    plt.title(_title)
    plt.legend(loc='best', frameon=False, prop={'size': 6})
    if filename is None:
        plt.show()
    else:
        plt.savefig("{}.png".format(filename))
        plt.clf()
