# This script plots histograms of r-statistic for in- and out-domain data, labeling them with the number of
# points included in each set, and their metric one score

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from package import gpr, io, rf, testhelper as th
import statistics


def prepareplot(res, sigma, number_of_bins):
    # Define input data -- divide by standard deviation
    model_errors = sigma
    abs_res = res

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
    
    return model_errors, abs_res, r_squared, slope, intercept, binned_model_errors, RMS_abs_res, xfit, yfit

######################## Make plots using function above ##############################


# load arrays of residuals and model errors saved in step 2
GPR_only_in_domain_RF_model_errors_hypercube = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_model_errors_hypercube.npy')
GPR_only_out_domain_RF_model_errors_hypercube = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_model_errors_hypercube.npy')
GPR_only_in_domain_RF_residuals_hypercube = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_residuals_hypercube.npy')
GPR_only_out_domain_RF_residuals_hypercube = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_residuals_hypercube.npy')
GPR_only_in_domain_RF_model_errors_hypershape = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_model_errors_hypershape.npy')
GPR_only_out_domain_RF_model_errors_hypershape = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_model_errors_hypershape.npy')
GPR_only_in_domain_RF_residuals_hypershape = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_in_domain_RF_residuals_hypershape.npy')
GPR_only_out_domain_RF_residuals_hypershape = np.load('Full_Method_Synthetic_noise/step_2_arrays/GPR_only_out_domain_RF_residuals_hypershape.npy')

print(len(GPR_only_in_domain_RF_model_errors_hypershape))
print(len(GPR_only_out_domain_RF_model_errors_hypershape))
X_test = np.load('Full_Method_Synthetic_noise/test_data/test_x_values_hypercube.npy')
print(len(X_test))

# Compute r-statistic ratios
GPR_only_in_hypercube = GPR_only_in_domain_RF_residuals_hypercube / GPR_only_in_domain_RF_model_errors_hypercube
GPR_only_out_hypercube = GPR_only_out_domain_RF_residuals_hypercube / GPR_only_out_domain_RF_model_errors_hypercube
GPR_only_in_hypershape = GPR_only_in_domain_RF_residuals_hypershape / GPR_only_in_domain_RF_model_errors_hypershape
GPR_only_out_hypershape = GPR_only_out_domain_RF_residuals_hypershape / GPR_only_out_domain_RF_model_errors_hypershape

# Prep data for input into function above
abs_res_in_hypercube = abs(GPR_only_in_domain_RF_residuals_hypercube)
abs_res_out_hypercube = abs(GPR_only_out_domain_RF_residuals_hypercube)

number_of_bins = 15
model_errors, abs_res, r_squared, slope, intercept, binned_model_errors, RMS_abs_res, xfit, yfit = prepareplot(abs_res_in_hypercube, GPR_only_in_domain_RF_model_errors_hypercube, number_of_bins)
model_errors2, abs_res2, r_squared2, slope2, intercept2, binned_model_errors2, RMS_abs_res2, xfit2, yfit2 = prepareplot(abs_res_out_hypercube, GPR_only_out_domain_RF_model_errors_hypercube, number_of_bins)

# function to compute metric one score
def metric_one(data):
	count = 0
	for i in range(0, len(data)):
		if abs(data[i]) > 1:
			count = count + 1
	print("count:")
	print(count)
	print("total:")
	print(len(data))
	score = count / (len(data) * 0.32)
	return score

# Compute scores
#GPR_in_score = metric_one(GPR_only_in)
#GPR_out_score = metric_one(GPR_only_out)

# Plot r-stat histograms

# GPR in-domain--hypercube
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- in-domain (hypercube test set)")
plt.text(1.5, 0.4, 'number of points = %d' %(len(GPR_only_in_hypercube)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_in_score))
plt.text(-5, 0.4, 'mean = %f' %(np.mean(GPR_only_in_hypercube)))
plt.text(-5, 0.35, 'stdev = %f' %(np.std(GPR_only_in_hypercube)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_in_hypercube, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_in_hypercube.png", dpi=300)
plt.clf()

# GPR out-domain--hypercube
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- out-domain (hypercube test set)")
plt.text(1.5, 0.35, 'number of points = %d' %(len(GPR_only_out_hypercube)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_out_score))
plt.text(-5, 0.35, 'mean = %f' %(np.mean(GPR_only_out_hypercube)))
plt.text(-5, 0.3, 'stdev = %f' %(np.std(GPR_only_out_hypercube)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_out_hypercube, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_out_hypercube.png", dpi=300)
plt.clf()

# GPR in-domain--hypershape
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- in-domain (hypershape test set)")
plt.text(1.5, 0.4, 'number of points = %d' %(len(GPR_only_in_hypershape)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_in_score))
plt.text(-5, 0.4, 'mean = %f' %(np.mean(GPR_only_in_hypershape)))
plt.text(-5, 0.35, 'stdev = %f' %(np.std(GPR_only_in_hypershape)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_in_hypershape, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_in_hypershape.png", dpi=300)
plt.clf()

# GPR out-domain--hypershape
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- out-domain (hypershape test set)")
plt.text(1.5, 0.5, 'number of points = %d' %(len(GPR_only_out_hypershape)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_out_score))
plt.text(-7.5, 0.5, 'mean = %f' %(np.mean(GPR_only_out_hypershape)))
plt.text(-7.5, 0.45, 'stdev = %f' %(np.std(GPR_only_out_hypershape)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_out_hypershape, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_out_hypershape.png", dpi=300)
plt.clf()

# RMS plots

# RMS plot for in-domain test data
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- in-domain only (Synthetic)")
plt.text(0.1,0.4,'r^2 = %f' %(r_squared))
plt.text(0.1,0.37, 'slope = %f' %(slope))
plt.text(0.1,0.34, 'y-intercept = %f' %(intercept))
plt.plot(binned_model_errors, RMS_abs_res, 'o', color='blue')
plt.plot(xfit, yfit)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/binned_scatter_plot_hypercube--in-domain.png", dpi = 300)
plt.clf()

# RMS plot for out-domain test data
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- out-domain only (Synthetic)")
plt.text(0.1,0.4,'r^2 = %f' %(r_squared2))
plt.text(0.1,0.37, 'slope = %f' %(slope2))
plt.text(0.1,0.34, 'y-intercept = %f' %(intercept2))
plt.plot(binned_model_errors2, RMS_abs_res2, 'o', color='blue')
plt.plot(xfit2, yfit2)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/binned_scatter_plot_hypercube--out-domain.png", dpi = 300)
plt.clf()