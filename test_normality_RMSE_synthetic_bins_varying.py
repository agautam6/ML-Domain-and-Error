import statistics
import numpy as np
from numpy import arange, meshgrid, array, round
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
import matplotlib.pyplot as plt
from tabulate import tabulate

it = 10
randomstate = None
gpr_thresholds_range = round(arange(0.1, 1.2, 0.1), 1)
rf_thresholds_range = round(arange(0.1, 1.2, 0.1), 1)
# normalityTests = ['RMSE', 'Shapiro-Wilk', 'DAgostino-Pearson']
normalityTests = ['RMSE']
defaults = {'RMSE': 1, 'Shapiro-Wilk': 0, 'DAgostino-Pearson': 0}
bin_sizes = [10, 50, 100, 200, 500]

rfslope = 0.256
rfintercept = 0.0184


# define training data size
training_num = 10000

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

# Train GPR
GPR = gpr.GPR()
GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=30)

# Train RF
RF = rf.RF()
RF.train_synth(X_train, y_train, std=standard_deviation)

# define test data size
test_num = 5000

# x-values: add or remove the *0.5 for each one
x0_test=np.random.rand(test_num)
x1_test=np.random.rand(test_num)*0.5
x2_test=np.random.rand(test_num)*0.5
x3_test=np.random.rand(test_num)*0.5
x4_test=np.random.rand(test_num)*0.5

X_test = [[x0_test[i], x1_test[i], x2_test[i], x3_test[i], x4_test[i]] for i in range(0,test_num)]

# y-value with friedman function
y_test = 30*np.sin(4*np.pi*x0_test*x1_test) + 20*(x2_test - 0.5)**2 + 10*x3_test + 5*x4_test

gpr_thresholds, rf_thresholds = meshgrid(gpr_thresholds_range, rf_thresholds_range)
accumulator = {(r, g, 1): [] for g in gpr_thresholds_range for r in rf_thresholds_range}
accumulator.update({(r, g, 0): [[], [], []] for g in gpr_thresholds_range for r in rf_thresholds_range})


gpr_pred, GPR_errors = GPR.predict(X_test, True)
rf_pred, RF_errors = RF.predict(X_test, True)
RF_errors = (rfslope * RF_errors) + rfintercept
for i_rf_thresholds in range(0, len(rf_thresholds_range)):
    for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
        print("GPR cutoff: " + str(i_gpr_thresholds) + " RF cutoff: " + str(i_rf_thresholds))
        gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        in_domain = accumulator[(rf_thresh, gpr_thresh, 1)]
        out_domain = accumulator[(rf_thresh, gpr_thresh, 0)]
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
                                              gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1]. \
                    append(residual_by_std / predicted_error if predicted_error else 0)


in_domain_norm_scores = {a: {b_i: [] for b_i in bin_sizes} for a in normalityTests}
out_domain_norm_scores = {a: {b_i: [] for b_i in bin_sizes} for a in normalityTests}
results = []

for i_rf_thresholds in range(0, len(rf_thresholds_range)):
    for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
        gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        cur_result = [rf_thresh, gpr_thresh]
        in_domain = accumulator[(rf_thresh, gpr_thresh, 1)]
        out_domain = accumulator[(rf_thresh, gpr_thresh, 0)]
        num_in_domain = len(in_domain)
        num_out_domain = len(out_domain[0]) + len(out_domain[1]) + len(out_domain[2])
        cur_result.append(num_in_domain)
        cur_result.append(num_out_domain)
        if num_in_domain is not 0:
            score = th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                             _xlabel='RF residual / RF predicted error',
                                             _ylabel='Normalized Counts',
                                             _title='in-domain Synthetic data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                     rf_thresh),
                                             filename='in_domain_Rstat_Modified_Friedman_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                      rf_thresh),
                                             _bincount=bin_sizes, _normalitytest=normalityTests, _range=(-50,50))
            for test in normalityTests:
                for b_i in bin_sizes:
                    in_domain_norm_scores[test][b_i].append(score[test][b_i])
        else:
            print('GPR Threshold = {} RF Threshold = {}, No points in-domain'.format(gpr_thresh, rf_thresh))
            for test in normalityTests:
                for b_i in bin_sizes:
                    in_domain_norm_scores[test][b_i].append(defaults[test])
        if num_out_domain is not 0:
            score = th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                             _xlabel='RF residual / RF predicted error',
                                             _ylabel='Normalized Counts',
                                             _title='out-domain Synthetic data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                      rf_thresh),
                                             filename='out_domain_Rstat_Modified_Friedman_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                       rf_thresh),
                                             _bincount=bin_sizes, _normalitytest=normalityTests, _range=(-50,50))
            for test in normalityTests:
                for b_i in bin_sizes:
                    out_domain_norm_scores[test][b_i].append(score[test][b_i])
        else:
            print('GPR Threshold = {} RF Threshold = {}, No points out-domain'.format(gpr_thresh, rf_thresh))
            for test in normalityTests:
                for b_i in bin_sizes:
                    out_domain_norm_scores[test][b_i].append(defaults[test])
        for test in normalityTests:
            for b_i in bin_sizes:
                cur_result.append(in_domain_norm_scores[test][b_i][-1])
                cur_result.append(out_domain_norm_scores[test][b_i][-1])
        results.append(cur_result)

for test in normalityTests:
    for b_i in bin_sizes:
        in_domain_norm_score_cur = array(in_domain_norm_scores[test][b_i]).reshape(
            (len(rf_thresholds_range), len(gpr_thresholds_range)))
        plt.contourf(gpr_thresholds, rf_thresholds, in_domain_norm_score_cur)
        plt.colorbar()
        plt.title('Synthetic In-Domain {} {} bins'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.savefig('Modified Friedman In-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

        out_domain_norm_score_cur = array(out_domain_norm_scores[test][b_i]).reshape(
            (len(rf_thresholds_range), len(gpr_thresholds_range)))
        plt.contourf(gpr_thresholds, rf_thresholds, out_domain_norm_score_cur)
        plt.colorbar()
        plt.title('Synthetic Out-Domain {} {} bins.png'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.savefig('Modified Friedman Out-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

fd = open('Normality_tests_Modified_Friedman_logs.txt', 'w')
log_headers = ["RF cutoff",
               "GPR cutoff",
               "Points in-domain",
               "Points out-domain"]
value_format = [".1f", ".1f", ".0f", ".0f"]
for testname in normalityTests:
    for b_i in bin_sizes:
        log_headers.append('In-Domain {} {} bins'.format(testname, b_i))
        log_headers.append('Out-Domain {} {} bins'.format(testname, b_i))
        value_format.append(".5f")
        value_format.append(".5f")
print(tabulate(results,
               headers=log_headers,
               tablefmt="github",
               floatfmt=value_format), file=fd)