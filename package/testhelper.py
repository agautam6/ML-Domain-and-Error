import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def GPR_plot(res, sigma, model_name, number_of_bins):
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

def RF_plot(res, sigma, model_name, number_of_bins):
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

    plt.show()



def predictdomain(GPR_error, RF_error):
    if GPR_error < 0.8 and RF_error < 0.8:
        return 1
    else:
        return 0
