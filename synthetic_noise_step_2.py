# This script calculates in and out of domain RF error estimates and residuals

import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
X_train = np.load('Full_Method_Synthetic_noise/training_data/training_x_values.npy')
y_train = np.load('Full_Method_Synthetic_noise/training_data/training_y_values.npy')
X_test_hypercube = np.load('Full_Method_Synthetic_noise/test_data/test_x_values_hypercube.npy')
y_test_hypercube = np.load('Full_Method_Synthetic_noise/test_data/test_y_values_hypercube.npy')
X_test_hypershape = np.load('Full_Method_Synthetic_noise/test_data/test_x_values_hypershape.npy')
y_test_hypershape = np.load('Full_Method_Synthetic_noise/test_data/test_y_values_hypershape.npy')

# define standard deviation of data set
standard_deviation = np.std(y_train)


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

# function to run cv splits on training set to determine domain cutoff values
def find_stats(X_values, y_values):
	RF_model_errors = np.asarray([])
	RF_residuals = np.asarray([])
	GPR_model_errors = np.asarray([])
	GPR_residuals = np.asarray([])
	# define cross-validation splits
	rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=91936274)
	#GPR
	ctr = 1
	for train_index, test_index in rkf.split(X_values):
		print("GPR: {}/20".format(ctr))
		ctr = ctr + 1
		X_train, X_test = X_values[train_index], X_values[test_index]
		y_train, y_test = y_values[train_index], y_values[test_index]
		GPR = gpr.GPR()
		GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=10)
		gpr_pred, gpr_errors = GPR.predict_no_divide(X_test, True)
		gpr_res = (y_test - gpr_pred) / standard_deviation
		gpr_errors = gpr_errors / standard_deviation
		GPR_model_errors = np.concatenate((GPR_model_errors, gpr_errors), axis=None)
		GPR_residuals = np.concatenate((GPR_residuals, gpr_res), axis=None)

	# define quantities to return
	GPR_model_error_std = np.std(GPR_model_errors)
	print("GPR standard deviation of model errors: {}".format(GPR_model_error_std))
	GPR_model_error_mean = np.mean(GPR_model_errors)
	print("GPR mean of model errors: {}".format(GPR_model_error_mean))

	cutoff = GPR_model_error_mean

	# RF
	print("finding rf scale factors")
	RF_model_errors = np.asarray([])
	RF_residuals = np.asarray([])
	for train_index, test_index in rkf.split(X_values):
		#print("RF: {}".format(ctr))
		#ctr = ctr + 1
		X_train, X_test = X_values[train_index], X_values[test_index]
		y_train, y_test = y_values[train_index], y_values[test_index]
		RF = rf.RF()
		RF.train_synth(X_train, y_train, std=standard_deviation)
		rf_pred, RF_errors = RF.predict_no_divide(X_test, True)
		rf_res = y_test - rf_pred
		RF_model_errors = np.concatenate((RF_model_errors, RF_errors), axis=None)
		RF_residuals = np.concatenate((RF_residuals, rf_res), axis=None)

	abs_residuals = abs(RF_residuals)

	res = np.asarray([])
	sigma = np.asarray([])
	# remove rf model errors and residuals that have gpr model error over cutoff
	for i in range(0,len(GPR_model_errors)):
		if GPR_model_errors[i] < cutoff:
			res = np.append(res, abs_residuals[i])
			sigma = np.append(sigma, RF_model_errors[i])



	return cutoff, res, sigma

# Initialize arrays to be saved at end of script
GPR_only_in_domain_RF_model_errors_hypercube = np.asarray([])
GPR_only_in_domain_RF_residuals_hypercube = np.asarray([])
GPR_only_out_domain_RF_model_errors_hypercube = np.asarray([])
GPR_only_out_domain_RF_residuals_hypercube = np.asarray([])
GPR_only_in_domain_RF_model_errors_hypershape = np.asarray([])
GPR_only_in_domain_RF_residuals_hypershape = np.asarray([])
GPR_only_out_domain_RF_model_errors_hypershape = np.asarray([])
GPR_only_out_domain_RF_residuals_hypershape = np.asarray([])


# Get domain cutoff values
gpr_cutoff, res, sigma = find_stats(X_train, y_train)
# Get model errors and residuals to make scaling plot
rfslope, rfintercept = rfscaling(res, sigma, standard_deviation, 15)
# Train RF
RF = rf.RF()
RF.train_synth(X_train, y_train, std=standard_deviation)
# hypercube predictions
rf_pred_hypercube, RF_errors_hypercube = RF.predict_no_divide(X_test_hypercube, True)
rf_errors_hypercube = (rfslope * (RF_errors_hypercube/standard_deviation)) + rfintercept
rf_res_hypercube = (y_test_hypercube - rf_pred_hypercube) / standard_deviation
# hypershape predictions
rf_pred_hypershape, RF_errors_hypershape = RF.predict_no_divide(X_test_hypershape, True)
rf_errors_hypershape = (rfslope * (RF_errors_hypershape/standard_deviation)) + rfintercept
rf_res_hypershape = (y_test_hypershape - rf_pred_hypershape) / standard_deviation
# Train GPR
GPR = gpr.GPR()
GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=30)
#hypercube predictions
gpr_pred_hypercube, gpr_errors_hypercube = GPR.predict_no_divide(X_test_hypercube, True)
gpr_res_hypercube = (y_test_hypercube - gpr_pred_hypercube) / standard_deviation
gpr_errors_hypercube = gpr_errors_hypercube / standard_deviation
# hypershape predictions
gpr_pred_hypershape, gpr_errors_hypershape = GPR.predict_no_divide(X_test_hypershape, True)
gpr_res_hypershape = (y_test_hypershape - gpr_pred_hypershape) / standard_deviation
gpr_errors_hypershape = gpr_errors_hypershape / standard_deviation
# Add RF residuals and model errors to appropriate arrays
for i in range(0, len(rf_errors_hypercube)):
	if gpr_errors_hypercube[i] < gpr_cutoff:
		GPR_only_in_domain_RF_residuals_hypercube = np.append(GPR_only_in_domain_RF_residuals_hypercube, rf_res_hypercube[i])
		GPR_only_in_domain_RF_model_errors_hypercube = np.append(GPR_only_in_domain_RF_model_errors_hypercube, rf_errors_hypercube[i])
	else:
		GPR_only_out_domain_RF_residuals_hypercube = np.append(GPR_only_out_domain_RF_residuals_hypercube, rf_res_hypercube[i])
		GPR_only_out_domain_RF_model_errors_hypercube = np.append(GPR_only_out_domain_RF_model_errors_hypercube, rf_errors_hypercube[i])
	if gpr_errors_hypershape[i] < gpr_cutoff:
		GPR_only_in_domain_RF_residuals_hypershape = np.append(GPR_only_in_domain_RF_residuals_hypershape, rf_res_hypershape[i])
		GPR_only_in_domain_RF_model_errors_hypershape = np.append(GPR_only_in_domain_RF_model_errors_hypershape, rf_errors_hypershape[i])
	else:
		GPR_only_out_domain_RF_residuals_hypershape = np.append(GPR_only_out_domain_RF_residuals_hypershape, rf_res_hypershape[i])
		GPR_only_out_domain_RF_model_errors_hypershape = np.append(GPR_only_out_domain_RF_model_errors_hypershape, rf_errors_hypershape[i])


# Save all arrays
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_model_errors_hypercube', GPR_only_in_domain_RF_model_errors_hypercube)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_residuals_hypercube', GPR_only_in_domain_RF_residuals_hypercube)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_model_errors_hypercube', GPR_only_out_domain_RF_model_errors_hypercube)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_residuals_hypercube', GPR_only_out_domain_RF_residuals_hypercube)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_model_errors_hypershape', GPR_only_in_domain_RF_model_errors_hypershape)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_residuals_hypershape', GPR_only_in_domain_RF_residuals_hypershape)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_model_errors_hypershape', GPR_only_out_domain_RF_model_errors_hypershape)
np.save('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_residuals_hypershape', GPR_only_out_domain_RF_residuals_hypershape)