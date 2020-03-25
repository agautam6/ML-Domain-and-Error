import statistics
import numpy as np
from numpy import arange, meshgrid, array, round
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
import matplotlib.pyplot as plt
from tabulate import tabulate

it = 10
randomstate = None
gpr_thresholds_range = arange(0.5, 1.2, 0.1)
rf_thresholds_range = arange(0.5, 1.2, 0.1)
#trainfile = 'data/Diffusion_Data_allfeatures.csv'
# rfslope = 0.65
# gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
#     .getGPRkernel()
gprsavedkernel = None

#data = io.importdata(trainfile)
#data = io.sanitizedata(data)
#X_all = data.iloc[:, 1:]
#y_all = data.iloc[:, 0]
#y_std = statistics.stdev(y_all.to_numpy(dtype=float))

# define training data size
training_num = 1000

# x-values: all uniformly distributed between 0 and 1
x0_train=np.random.rand(training_num)*0.5
x1_train=np.random.rand(training_num)*0.5
x2_train=np.random.rand(training_num)*0.5
x3_train=np.random.rand(training_num)*0.5
x4_train=np.random.rand(training_num)*0.5

X_train = [[x0_train[i], x1_train[i], x2_train[i], x3_train[i], x4_train[i]] for i in range(0,training_num)]

# y-value with friedman function
y_train = 30*np.sin(4*np.pi*x0_train*x1_train) + 20*(x2_train - 0.5)**2 + 10*x3_train + 5*x4_train

# Define standard deviation of training data
standard_deviation = np.std(y_train)

# Define values for scaling RF predicted errors
rf_slope = 0.255
rf_intercept = 0.0183

# Train GPR
GPR = gpr.GPR()
GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=30)

# Train RF
RF = rf.RF()
RF.train_synth(X_train, y_train, std=standard_deviation)

# define test data size
test_num = 1000

# state which variable will be varied
k = 4

# x-values: add or remove the *0.5 for each one
x0_test=np.random.rand(test_num)*0.5
x1_test=np.random.rand(test_num)*0.5
x2_test=np.random.rand(test_num)*0.5
x3_test=np.random.rand(test_num)*0.5
x4_test=np.random.rand(test_num)

X_test = [[x0_test[i], x1_test[i], x2_test[i], x3_test[i], x4_test[i]] for i in range(0,test_num)]

# y-value with friedman function
y_test = 30*np.sin(4*np.pi*x0_test*x1_test) + 20*(x2_test - 0.5)**2 + 10*x3_test + 5*x4_test


gpr_thresholds, rf_thresholds = meshgrid(gpr_thresholds_range, rf_thresholds_range)
in_domain_norm_score_RMS = []
out_domain_norm_score_RMS = []
results = []

for i_rf_thresholds in range(0, len(rf_thresholds_range)):
    for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
        gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        print("Begin: GPR Threshold = {} RF Threshold = {}".format(gpr_thresh, rf_thresh))
        in_domain = []
        out_domain = [[], [], []]
        cur_result = [rf_thresh, gpr_thresh]
        # count = 1
        # rs = ShuffleSplit(n_splits=it, test_size=.3, random_state=randomstate)
        gpr_pred, GPR_errors = GPR.predict(X_test, True)
        rf_pred, RF_errors = RF.predict(X_test, True)
        RF_errors = rf_slope * RF_errors + rf_intercept
        predictions = [th.predictdomain(GPR_errors[i], RF_errors[i],
                                        gpr_threshold=gpr_thresh, rf_threshold=rf_thresh)
                       for i in range(0, len(X_test))]
        for i in range(0, len(X_test)):
            residual_by_std = (y_test[i] - rf_pred[i]) / standard_deviation
            predicted_error = RF_errors[i]
            if predictions[i] is 1:
                in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
            else:
                out_domain[th.getcontribution(GPR_errors[i], RF_errors[i],
                                              gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1].\
                    append(residual_by_std / predicted_error if predicted_error else 0)
        num_in_domain = len(in_domain) if len(in_domain) is not 0 else 0
        num_out_domain = len(out_domain[0]) + len(out_domain[1]) + len(out_domain[2]) \
            if len(out_domain[0]) + len(out_domain[1]) + len(out_domain[2]) is not 0 else 0
        cur_result.append(num_in_domain)
        cur_result.append(num_out_domain)
        if num_in_domain is not 0:
            score = th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                             _xlabel='RF residual / RF predicted error',
                                             _ylabel='Normalized Counts',
                                             _title='in-domain synthetic data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                     rf_thresh),
                                             filename='synthetic_domain_test_x4/in_domain_Rstat_modified_Friedman_x4_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                      rf_thresh),
                                             _bincount=50, _normalitytest=['RMSE'])
            in_domain_norm_score_RMS.append(score[0])
        else:
            print('GPR Threshold = {} RF Threshold = {}, No points in-domain'.format(gpr_thresh, rf_thresh))
            in_domain_norm_score_RMS.append(1)
        if num_out_domain is not 0:
            score = th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                             _xlabel='RF residual / RF predicted error',
                                             _ylabel='Normalized Counts',
                                             _title='out-domain synthetic data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                      rf_thresh),
                                             filename='synthetic_domain_test_x4/out_domain_Rstat_modified_Friedman_x4_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                       rf_thresh),
                                             _bincount=50, _normalitytest=['RMSE'])
            out_domain_norm_score_RMS.append(score[0])
        else:
            print('GPR Threshold = {} RF Threshold = {}, No points out-domain'.format(gpr_thresh, rf_thresh))
            out_domain_norm_score_RMS.append(1)
        cur_result.append(in_domain_norm_score_RMS[-1])
        cur_result.append(out_domain_norm_score_RMS[-1])
        results.append(cur_result)
        print("End: GPR Threshold = {} RF Threshold = {}".format(gpr_thresh, rf_thresh))


in_domain_norm_score_RMS = array(in_domain_norm_score_RMS).reshape(
    (len(rf_thresholds_range), len(gpr_thresholds_range)))
plt.contourf(gpr_thresholds, rf_thresholds, in_domain_norm_score_RMS, cmap='RdYlGn_r')
plt.colorbar()
plt.title('In-Domain Normality RMSE Modified Friedman x4')
plt.xlabel('GPR cutoff')
plt.ylabel('RF cutoff')
plt.savefig('synthetic_domain_test_x4/In-Domain Normality RMSE Contour Plot Modified Friedman Synthetic data x4 varying.png')
plt.clf()

out_domain_norm_score_RMS = array(out_domain_norm_score_RMS).reshape(
    (len(rf_thresholds_range), len(gpr_thresholds_range)))
plt.contourf(gpr_thresholds, rf_thresholds, out_domain_norm_score_RMS, cmap='RdYlGn_r')
plt.colorbar()
plt.title('Out-Domain Normality RMSE Modified Friedman x4')
plt.xlabel('GPR cutoff')
plt.ylabel('RF cutoff')
plt.savefig('synthetic_domain_test_x4/Out-Domain Normality RMSE Contour Plot Modified Friedman Synthetic data x4 varying.png')
plt.clf()

fd = open('synthetic_domain_test_x4/Normality_RMSE_Modified_Friedman_x4_data_logs.txt', 'w')
print(tabulate(results,
               headers=["RF cutoff",
                        "GPR cutoff",
                        "Points in-domain",
                        "Points out-domain",
                        "In-domain Normality RMSE",
                        "Out-domain Normality RMSE"],
               tablefmt="github",
               floatfmt=[".1f", ".1f", ".0f", ".0f", ".5f", ".5f"]), file=fd)