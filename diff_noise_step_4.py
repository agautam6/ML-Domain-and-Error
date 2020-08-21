# This script runs 5-fold CV to generate scaled and unscaled random-forest error estimates, along
# with their corresponding residuals. These will be used to generate RMS and r-stat plots for the patent
# application.

import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
X_train_total = np.load('Full_Method_Diffusion_noise/data/all_x_values.npy')
y_train_total = np.load('Full_Method_Diffusion_noise/data/all_y_values.npy')

# define standard deviation of data set
standard_deviation = np.std(y_train_total)

# initialize counter
outerctr = 1

# function to calculate rf scaling factors
def rfscaling(res, sigma, stdev, number_of_bins):
    # Define input data -- divide by standard deviation
    model_errors = sigma / stdev
    abs_res = res / stdev

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
    for i in range(1,number_of_bins + 1):
        if i in digitized:
            weights.append(np.count_nonzero(digitized == i))
    
    # Calculate RMS of the absolute residuals
    RMS_abs_res = [np.sqrt((abs_res[digitized == bins_present[i]] ** 2).mean()) for i in range(0, len(bins_present))]

    # Set the x-values to the midpoint of each bin
    bin_width = bins[1]-bins[0]
    binned_model_errors = np.zeros(len(bins_present))
    for i in range(0, len(bins_present)):
        curr_bin = bins_present[i]
        binned_model_errors[i] = bins[curr_bin-1] + bin_width/2

    # Fit a line to the data
    model = LinearRegression(fit_intercept=True)
    model.fit(binned_model_errors[:, np.newaxis],
                  RMS_abs_res, sample_weight=weights)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    xfit = binned_model_errors
    yfit = model.predict(xfit[:, np.newaxis])

    # Calculate r^2 value
    r_squared = r2_score(RMS_abs_res, yfit, sample_weight=weights)
    # Calculate slope
    slope = model.coef_
    # Calculate y-intercept
    intercept = model.intercept_

    print("rf slope: {}".format(slope))
    print("rf intercept: {}".format(intercept))
    
    return slope, intercept

def find_stats(X_values, y_values, stdev):
	# define cross-validation splits
	rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=91936274)
	# RF
	print("finding rf scale factors")
	RF_model_errors = np.asarray([])
	RF_resid = np.asarray([])
	for train_index, test_index in rkf.split(X_values):
		#print("RF: {}".format(ctr))
		#ctr = ctr + 1
		X_train, X_test = X_values[train_index], X_values[test_index]
		y_train, y_test = y_values[train_index], y_values[test_index]
		RF = rf.RF()
		RF.train_synth(X_train, y_train, std=stdev)
		rf_pred, RF_errors = RF.predict_no_divide(X_test, True)
		rf_res = y_test - rf_pred
		RF_model_errors = np.concatenate((RF_model_errors, RF_errors), axis=None)
		RF_resid = np.concatenate((RF_resid, rf_res), axis=None)

	abs_residuals = abs(RF_resid)

	return abs_residuals, RF_model_errors



RF_unscaled_error_bars = np.asarray([])
RF_scaled_error_bars = np.asarray([])
RF_residuals = np.asarray([])

# Do repeated kfold cross-validation for GPR and RF
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=59173474)

# RF
for train_index, test_index in rkf.split(X_train_total):
	# Define x and y values
	X_train, X_test = X_train_total[train_index], X_train_total[test_index]
	y_train, y_test = y_train_total[train_index], y_train_total[test_index]
	# Get standard deviation of training set
	std = np.std(y_train)
	# Get stats for finding scale factors
	res, sigma = find_stats(X_train, y_train, std)
	# Get model errors and residuals to make scaling plot
	rfslope, rfintercept = rfscaling(res, sigma, std, 15)
	# Train RF
	RF = rf.RF()
	RF.train_synth(X_train, y_train, std=standard_deviation)
	rf_pred, RF_errors = RF.predict_no_divide(X_test, True)
	rf_errors_scaled = (rfslope * (RF_errors/std)) + rfintercept
	rf_errors_unscaled = RF_errors / std
	rf_res = (y_test - rf_pred) / std
	for i in range(0, len(rf_res)):
		RF_unscaled_error_bars = np.append(RF_unscaled_error_bars, rf_errors_unscaled[i])
		RF_scaled_error_bars = np.append(RF_scaled_error_bars, rf_errors_scaled[i])
		RF_residuals = np.append(RF_residuals, rf_res[i])

np.save('Full_Method_Diffusion_noise/patent_plots/unscaled_rf_error_estimates', RF_unscaled_error_bars)
np.save('Full_Method_Diffusion_noise/patent_plots/scaled_rf_error_estimates', RF_scaled_error_bars)
np.save('Full_Method_Diffusion_noise/patent_plots/rf_residuals', RF_residuals)
