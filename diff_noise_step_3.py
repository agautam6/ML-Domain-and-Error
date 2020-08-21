# This script plots histograms of r-statistic for in- and out-domain data, labeling them with the number of
# points included in each set, and their metric one score

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# load arrays of residuals and model errors saved in step 3
GPR_only_in_domain_RF_model_errors = np.load('Full_Method_Diffusion_noise/step_2_arrays/GPR_only_in_domain_RF_model_errors.npy')
GPR_only_out_domain_RF_model_errors = np.load('Full_Method_Diffusion_noise/step_2_arrays/GPR_only_out_domain_RF_model_errors.npy')
GPR_only_in_domain_RF_residuals = np.load('Full_Method_Diffusion_noise/step_2_arrays/GPR_only_in_domain_RF_residuals.npy')
GPR_only_out_domain_RF_residuals = np.load('Full_Method_Diffusion_noise/step_2_arrays/GPR_only_out_domain_RF_residuals.npy')

# Compute r-statistic ratios
GPR_only_in = GPR_only_in_domain_RF_residuals / GPR_only_in_domain_RF_model_errors
GPR_only_out = GPR_only_out_domain_RF_residuals / GPR_only_out_domain_RF_model_errors

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

# GPR in-domain
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- in-domain (GPR cutoff only)")
plt.text(1.5, 0.4, 'number of points = %d' %(len(GPR_only_in)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_in_score))
plt.text(-5, 0.4, 'mean = %f' %(np.mean(GPR_only_in)))
plt.text(-5, 0.35, 'stdev = %f' %(np.std(GPR_only_in)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_in, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Diffusion_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_in.png")
plt.clf()

# GPR out-domain
plt.xlabel("RF residuals / RF model errors")
plt.ylabel("Counts")
plt.title("Residual / Model error -- out-domain (GPR cutoff only)")
plt.text(1.5, 0.5, 'number of points = %d' %(len(GPR_only_out)))
#plt.text(1.5, 0.45, 'metric one score = %f' %(GPR_out_score))
plt.text(-7.5, 0.5, 'mean = %f' %(np.mean(GPR_only_out)))
plt.text(-7.5, 0.45, 'stdev = %f' %(np.std(GPR_only_out)))
x = np.linspace(-5, 5, 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='Gaussian mu: 0 std: 1')
plt.hist(GPR_only_out, bins=50, color='blue', edgecolor='black', density=True)
plt.savefig("Full_Method_Diffusion_noise/in_and_out_domain_rstat_plots/new-scaling_GPR_only_out.png")
plt.clf()