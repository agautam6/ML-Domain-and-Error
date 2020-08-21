# This script generates unscaled and scaled RMS and r-stat plots using the data generated in step 4

import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats


#################### Define function to output needed data #########################

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

sigma_test_unscaled_hypercube = np.load('Full_Method_Synthetic_noise/patent_plots/unscaled_rf_error_estimates_hypercube.npy')
sigma_test_scaled_hypercube = np.load('Full_Method_Synthetic_noise/patent_plots/scaled_rf_error_estimates_hypercube.npy')
y_res_test_hypercube = np.load('Full_Method_Synthetic_noise/patent_plots/rf_residuals_hypercube.npy')
sigma_test_unscaled_hypershape = np.load('Full_Method_Synthetic_noise/patent_plots/unscaled_rf_error_estimates_hypershape.npy')
sigma_test_scaled_hypershape = np.load('Full_Method_Synthetic_noise/patent_plots/scaled_rf_error_estimates_hypershape.npy')
y_res_test_hypershape = np.load('Full_Method_Synthetic_noise/patent_plots/rf_residuals_hypershape.npy')

abs_y_res_test_hypercube = abs(y_res_test_hypercube)
abs_y_res_test_hypershape = abs(y_res_test_hypershape)

number_of_bins = 15

model_errors1, abs_res1, r_squared1, slope1, intercept1, binned_model_errors1, RMS_abs_res1, xfit1, yfit1 = prepareplot(abs_y_res_test_hypercube, sigma_test_unscaled_hypercube, number_of_bins)

model_errors2, abs_res2, r_squared2, slope2, intercept2, binned_model_errors2, RMS_abs_res2, xfit2, yfit2 = prepareplot(abs_y_res_test_hypercube, sigma_test_scaled_hypercube, number_of_bins)

model_errors3, abs_res3, r_squared3, slope3, intercept3, binned_model_errors3, RMS_abs_res3, xfit3, yfit3 = prepareplot(abs_y_res_test_hypershape, sigma_test_unscaled_hypershape, number_of_bins)

model_errors4, abs_res4, r_squared4, slope4, intercept4, binned_model_errors4, RMS_abs_res4, xfit4, yfit4 = prepareplot(abs_y_res_test_hypershape, sigma_test_scaled_hypershape, number_of_bins)

gaussian_x = np.linspace(-5, 5, 1000)


##################### Hypercube plots ##################

# TEST (hypercube) binned RMS plot
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- NOT SCALED (hypercube)")
plt.text(0.4,1.15,'r^2 = %f' %(r_squared1))
plt.text(0.4,1.12, 'slope = %f' %(slope1))
plt.text(0.4,1.09, 'y-intercept = %f' %(intercept1))
plt.plot(binned_model_errors1, RMS_abs_res1, 'o', color='blue')
plt.plot(xfit1, yfit1)
plt.savefig("Full_Method_Synthetic_noise/patent_plots/binned_scatter_plot--unscaled_test_hypercube.png")
plt.clf()

# TEST (hypercube) binned RMS plot
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- SCALED (hypercube)")
plt.text(0.922,1.20,'r^2 = %f' %(r_squared2))
plt.text(0.922,1.17, 'slope = %f' %(slope2))
plt.text(0.922,1.14, 'y-intercept = %f' %(intercept2))
plt.plot(binned_model_errors2, RMS_abs_res2, 'o', color='blue')
plt.plot(xfit2, yfit2)
plt.savefig("Full_Method_Synthetic_noise/patent_plots/binned_scatter_plot--scaled_test_hypercube.png")
plt.clf()

# TEST r-stat plots (10,000 hypercube)
#unscaled
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Normalized Counts")
plt.title("Residual / Model error -- NOT SCALED (hypercube)")
plt.text(-5, 0.35, 'mean = %f' %(np.mean(y_res_test_hypercube/sigma_test_unscaled_hypercube)))
plt.text(-5, 0.32, 'std = %f' %(np.std(y_res_test_hypercube/sigma_test_unscaled_hypercube)))
plt.hist((y_res_test_hypercube)/(sigma_test_unscaled_hypercube), bins=30, color='blue', edgecolor='black', density=True)
plt.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='orange')
plt.savefig("Full_Method_Synthetic_noise/patent_plots/unscaled_rstat_histogram_test_hypercube.png")
plt.clf()
#scaled
plt.xlabel("RF residuals / (RF model errors * slope + intercept)")
plt.ylabel("Normalized Counts")
plt.title("Residual / Model error -- SCALED (hypercube)")
plt.text(-5, 0.35, 'mean = %f' %(np.mean((y_res_test_hypercube)/((sigma_test_scaled_hypercube)))))
plt.text(-5, 0.32, 'std = %f' %(np.std((y_res_test_hypercube)/((sigma_test_scaled_hypercube)))))
plt.hist((y_res_test_hypercube)/((sigma_test_scaled_hypercube)), bins=30, color='blue', edgecolor='black', density=True)
plt.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='orange')
plt.savefig("Full_Method_Synthetic_noise/patent_plots/scaled_rstat_histogram_test_hypercube.png")
plt.clf()

################ Hypershape plots #####################

# TEST (hypershape) binned RMS plot
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- NOT SCALED (hypershape)")
plt.text(0.8,0.7,'r^2 = %f' %(r_squared3))
plt.text(0.8,0.6, 'slope = %f' %(slope3))
plt.text(0.8,0.5, 'y-intercept = %f' %(intercept3))
plt.plot(binned_model_errors3, RMS_abs_res3, 'o', color='blue')
plt.plot(xfit3, yfit3)
plt.savefig("Full_Method_Synthetic_noise/patent_plots/binned_scatter_plot--unscaled_test_hypershape.png")
plt.clf()

# TEST (hypershape) binned RMS plot
plt.xlabel("RF model errors / dataset stdev")
plt.ylabel("RF RMS Absolute residuals / dataset stdev")
plt.title("RF RMS Absolute Residuals vs. Model Errors -- SCALED (hypershape)")
plt.text(0.921,0.7,'r^2 = %f' %(r_squared4))
plt.text(0.921,0.6, 'slope = %f' %(slope4))
plt.text(0.921,0.5, 'y-intercept = %f' %(intercept4))
plt.plot(binned_model_errors4, RMS_abs_res4, 'o', color='blue')
plt.plot(xfit4, yfit4)
plt.savefig("Full_Method_Synthetic_noise/patent_plots/binned_scatter_plot--scaled_test_hypershape.png")
plt.clf()

# TEST r-stat plots (10,000 hypershape)
#unscaled
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Normalized Counts")
plt.title("Residual / Model error -- NOT SCALED (hypershape)")
plt.text(-5, 0.35, 'mean = %f' %(np.mean(y_res_test_hypershape/sigma_test_unscaled_hypershape)))
plt.text(-5, 0.32, 'std = %f' %(np.std(y_res_test_hypershape/sigma_test_unscaled_hypershape)))
plt.hist((y_res_test_hypershape)/(sigma_test_unscaled_hypershape), bins=30, color='blue', edgecolor='black', density=True)
plt.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='orange')
plt.savefig("Full_Method_Synthetic_noise/patent_plots/unscaled_rstat_histogram_test_hypershape.png")
plt.clf()
#scaled
plt.xlabel("RF residuals / (RF model errors * slope + intercept)")
plt.ylabel("Normalized Counts")
plt.title("Residual / Model error -- SCALED (hypershape)")
plt.text(-5, 0.35, 'mean = %f' %(np.mean((y_res_test_hypershape)/((sigma_test_scaled_hypershape)))))
plt.text(-5, 0.32, 'std = %f' %(np.std((y_res_test_hypershape)/((sigma_test_scaled_hypershape)))))
plt.hist((y_res_test_hypershape)/((sigma_test_scaled_hypershape)), bins=30, color='blue', edgecolor='black', density=True)
plt.plot(gaussian_x, stats.norm.pdf(gaussian_x, 0, 1), label='Gaussian mu: 0 std: 1', color='orange')
plt.savefig("Full_Method_Synthetic_noise/patent_plots/scaled_rstat_histogram_test_hypershape.png")
plt.clf()



