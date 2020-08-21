# This script plots histograms of r-statistic for in- and out-domain data, labeling them with the number of
# points included in each set, and their metric one score

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


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
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_in_hypercube.png")
plt.clf()

# GPR out-domain--hypercube
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- out-domain (hypercube test set)")
plt.text(1.5, 0.5, 'number of points = %d' %(len(GPR_only_out_hypercube)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_out_score))
plt.text(-7.5, 0.5, 'mean = %f' %(np.mean(GPR_only_out_hypercube)))
plt.text(-7.5, 0.45, 'stdev = %f' %(np.std(GPR_only_out_hypercube)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_out_hypercube, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_out_hypercube.png")
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
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_in_hypershape.png")
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
plt.savefig("Full_Method_Synthetic_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_out_hypershape.png")
plt.clf()