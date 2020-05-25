from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pkg_resources import resource_stream
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

normality_benchmark = \
    load(
        resource_stream(__name__, 'resources/normality-benchmarks/normality_benchmark_rmse_averaged_04-09-20_22-54-55'))
log_normality_benchmark = \
    load(resource_stream(__name__, 'resources/normality-benchmarks/normality_benchmark_negative_log_rmse_averaged_04'
                                   '-09-20_22-54-56'))
defaults = {'RMSE': 1, 'Shapiro-Wilk': 0, 'DAgostino-Pearson': 0, 'Normalized-RMSE': 1,
            'Log-RMSE': 0, 'Normalized-Log-RMSE': 0, 'MetricOne': -10, 'MetricTwo': -10}


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

    # Create array of weights based on counts in each bin
    weights = []
    for i in range(1, number_of_bins + 1):
        if i in digitized:
            weights.append(np.count_nonzero(digitized == i))

    # Calculate RMS of the absolute residuals
    RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]

    # Set the x-values to the midpoint of each bin
    bin_width = bins[1] - bins[0]
    binned_model_errors = np.zeros(len(bins_present))
    for i in range(0, len(bins_present)):
        curr_bin = bins_present[i]
        binned_model_errors[i] = bins[curr_bin - 1] + bin_width / 2

    # Fit a line to the data
    model = LinearRegression(fit_intercept=True)
    model.fit(binned_model_errors[:, np.newaxis],
              RMS_abs_res,
              sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    xfit = binned_model_errors
    yfit = model.predict(xfit[:, np.newaxis])

    # Calculate r^2 value
    r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
    # Calculate slope
    slope = model.coef_
    # Calculate y-intercept
    intercept = model.intercept_

    # Create RMS scatter plot
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("%s RMS Absolute residuals / dataset stdev" % (model_name))
    plt.ylim(0, 1)
    plt.title("%s RMS Absolute Residuals vs. Model Errors" % (model_name))
    plt.text(0.2, 0.9, 'r^2 = %f' % (r_squared))
    plt.text(0.2, 0.8, 'slope = %f' % (slope))
    plt.text(0.2, 0.7, 'y-intercept = %f' % (intercept))
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

    # Create array of weights based on counts in each bin
    weights = []
    for i in range(1, number_of_bins + 1):
        if i in digitized:
            weights.append(np.count_nonzero(digitized == i))

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
    model = LinearRegression(fit_intercept=True)
    model.fit(binned_model_errors[0:cutoff_bin, np.newaxis],
              RMS_abs_res[0:cutoff_bin], sample_weight=weights[
                                                       0:cutoff_bin])  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    xfit = binned_model_errors[0:cutoff_bin]
    yfit = model.predict(xfit[:, np.newaxis])

    # Calculate r^2 value
    r_squared = r2_score(RMS_abs_res[0:cutoff_bin], yfit, sample_weight=weights[0:cutoff_bin])
    # Calculate slope
    slope = model.coef_
    # Calculate y-intercept
    intercept = model.intercept_

    # Create RMS scatter plot
    plt.xlabel("%s model errors / dataset stdev" % (model_name))
    plt.ylabel("%s RMS Absolute residuals / dataset stdev" % (model_name))
    plt.ylim(0, 1)
    plt.title("%s RMS Absolute Residuals vs. Model Errors" % (model_name))
    plt.text(0.2, 0.9, 'r^2 = %f' % (r_squared))
    plt.text(0.2, 0.8, 'slope = %f' % (slope))
    plt.text(0.2, 0.7, 'y-intercept = %f' % (intercept))
    plt.plot(binned_model_errors[0:cutoff_bin], RMS_abs_res[0:cutoff_bin], 'o', color='blue')
    plt.plot(binned_model_errors[cutoff_bin:], RMS_abs_res[cutoff_bin:], 'o', color='red')
    plt.plot(xfit, yfit);

    if filename is None:
        plt.show()
    else:
        plt.savefig("{}_plot3.png".format(filename))
        plt.clf()


def predictdomain(GPR_error, RF_error, gpr_threshold=0.8, rf_threshold=0.8):
    if GPR_error < gpr_threshold and RF_error < rf_threshold:
        return 1
    else:
        return 0


def getcontribution(GPR_error, RF_error, gpr_threshold=0.8, rf_threshold=0.8):
    if GPR_error < gpr_threshold and RF_error < rf_threshold:
        return 0
    elif RF_error < rf_threshold and gpr_threshold <= GPR_error:
        return 1
    elif GPR_error < gpr_threshold and rf_threshold <= RF_error:
        return 2
    else:
        return 3


def getMetricOne(data):
    outside = 0
    total = len(data)
    for i in range(0, total):
        if data[i] > 1 or data[i] < -1:
            outside = outside + 1
    if (total * 0.32) >= 20:
        return outside / (total * 0.32)
    else:
        return -10


def getMetricTwo(data):
    outside = 0
    total = len(data)
    for i in range(0, total):
        if data[i] > 2 or data[i] < -2:
            outside = outside + 1
    if (total * 0.05) >= 20:
        return outside / (total * 0.05)
    else:
        return -10


def getLogRMSnormalityscore(counts, bins):
    return np.log10(getRMSnormalityscore(counts, bins))


def getRMSnormalityscore(counts, bins):
    return mean_squared_error(stats.norm.cdf(bins[1:]) - stats.norm.cdf(bins[:-1]),
                              np.multiply(counts, (bins[1] - bins[0])))


def getShapiroWilkScore(x):
    return 0 if len(x) < 3 else stats.shapiro(x)[1]


def getDAgostinoPearsonScore(x):
    return 0 if len(x) < 8 else stats.normaltest(x)[1]


# Expects non-empty 'data'
def plotrstatwithgaussian(data, _stacked=True, _label=None,
                          _savePlot=(True, '.', '.', 'plot.png'),
                          filename=None,
                          _xlabel="", _ylabel="", _range=(-5, 5), _bincount=[10], _title="", _normalitytest=None
                          , temp_file=None):
    onelist = data
    if not isinstance(data[0], list):  # checking for multiple data sets with only 1st element instead of all()
        total = len(data)
    else:
        onelist = [val for sublist in data for val in sublist]
        total = sum([len(i) for i in data])
    normalityscore = {a: {b_i: [] for b_i in _bincount} for a in _normalitytest}
    if total > 0:
        (mu, sigma) = stats.norm.fit(onelist)
    for b_i in _bincount:
        if total > 0:
            n, bins, patches = plt.hist(data, density=True, label=_label, stacked=_stacked, bins=b_i, range=_range)
            if isinstance(n[0], np.ndarray):
                n = [sum(i) for i in zip(*n)]
            if total > 0:
                # _savePlot[0] is True and \
                x = np.linspace(-5, 5, 1000)
                plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
                plt.plot(x, stats.norm.pdf(x, mu, sigma),
                         label='Gaussian mu: {} std: {}'.format(round(mu, 2), round(sigma, 2)))
                plt.ylabel(_ylabel)
                plt.xlabel(_xlabel)
                plt.title(_title + " ({} points {} bins)".format(total, b_i))
                plt.legend(loc='best', frameon=False, prop={'size': 6})

                z = filename + temp_file + ".png"

                if z is None:
                    plt.show()
                else:
                    plt.savefig(z)

                # if _savePlot[0] is False:
                #     plt.show()
                # else:
                #     Path("{}/{}-bins/{}".format(_savePlot[1], b_i, _savePlot[2])).mkdir(parents=True, exist_ok=True)
                #     plt.savefig(
                #         "{}/{}-bins/{}/{}_{}_bins.png".format(_savePlot[1], b_i, _savePlot[2], _savePlot[3], b_i))
            plt.clf()
        if _normalitytest is not None:
            for i in _normalitytest:
                if total < 20:
                    normalityscore[i][b_i] = defaults[i]
                    continue
                if i == 'Normalized-Log-RMSE':
                    normalityscore[i][b_i] = (-np.log(log_normality_benchmark[b_i][len(onelist) - 1])) / np.log(
                        10) - getLogRMSnormalityscore(n, bins)
                elif i == 'Log-RMSE':
                    normalityscore[i][b_i] = getLogRMSnormalityscore(n, bins)
                elif i == 'Normalized-RMSE':
                    normalityscore[i][b_i] = (
                            getRMSnormalityscore(n, bins) / normality_benchmark[b_i][len(onelist) - 1])
                elif i == 'RMSE':
                    normalityscore[i][b_i] = getRMSnormalityscore(n, bins)
                elif i == 'Shapiro-Wilk':
                    normalityscore[i][b_i] = getShapiroWilkScore(onelist)
                elif i == 'DAgostino-Pearson':
                    normalityscore[i][b_i] = getDAgostinoPearsonScore(onelist)
                elif i == 'MetricOne':
                    normalityscore[i][b_i] = getMetricOne(onelist)
                elif i == 'MetricTwo':
                    normalityscore[i][b_i] = getMetricTwo(onelist)
    return normalityscore
