import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tabulate import tabulate

from package import gpr, io, rf, testhelper as th

# Initialization
gpr_thresholds_range = np.round(np.arange(0.5, 1.2, 0.1), 1)
rf_thresholds_range = np.round(np.arange(0.5, 1.2, 0.1), 1)
include_INF = True
normalityTests = ['RMSE', 'Normalized-RMSE', 'Log-RMSE', 'Normalized-Log-RMSE']
bin_sizes = [10, 50, 100, 200, 500]
contour_plot_same_scale = False
make_counts_plot = True

# Resources
trainfile = '../data/Diffusion_Data_allfeatures.csv'
# trainfile = '../data/temp.csv'
rfslope = 0.919216
rfintercept = -0.025370
gprsavedkernel = io.loadmodelobj('../models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

# Data
data = io.importdata(trainfile)
groups = data['Material compositions 1'].values
data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
y_std = statistics.stdev(Y.to_numpy(dtype=float))

# For Infinite Cutoffs
INF = np.inf
if include_INF:
    gpr_thresholds_range = np.append(gpr_thresholds_range, INF)
    rf_thresholds_range = np.append(rf_thresholds_range, INF)
gpr_thresholds, rf_thresholds = np.meshgrid(gpr_thresholds_range, rf_thresholds_range)
accumulator = {(r, g, 1): [] for g in gpr_thresholds_range for r in rf_thresholds_range}
accumulator.update({(r, g, 0): [[], [], []] for g in gpr_thresholds_range for r in rf_thresholds_range})

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
in_domain = []
out_domain = [[], [], []]
alreadyDone = []
count = 0

path = os.path.abspath('..') + "/domain results/Normalized RMSE/Diffusion LOG Pair/INF/"


def checkAlreadyDone(element, alreadylist):
    for x in alreadylist:
        if element == x:
            return True
    return False


# # Getting the values computed previously:
# if os.path.isfile('LOG_Pair_values.pkl'):
#     with open('LOG_Pair_values.pkl', 'rb') as f:
#         in_domain, out_domain = pickle.load(f)
#
# else:
for train_index, test_index in rfk.split(X, Y, groups):
    X_train_1, X_test_1 = X.iloc[train_index], X.iloc[test_index]
    y_train_1, y_test_1 = Y.iloc[train_index], Y.iloc[test_index]

    groups2 = np.delete(groups, test_index)
    testGroup = np.delete(groups, train_index)

    for train_index_2, test_index_2 in rfk.split(X_train_1, y_train_1, groups2):
        X_train, X_test = X.iloc[train_index_2], X.iloc[test_index_2]
        y_train, y_test = Y.iloc[train_index_2], Y.iloc[test_index_2]

        testGroup2 = np.delete(groups2, train_index_2)

        if checkAlreadyDone(testGroup2[0], alreadyDone):
            continue

        frames = [X_test_1, X_test]
        twoTest = pd.concat(frames)

        yTest = [y_test_1, y_test]
        yFrames = pd.concat(yTest)

        testFinal = np.concatenate((testGroup, testGroup2))

        RF = rf.RF()
        RF.train(X_train, y_train, std=y_std)

        GPR = gpr.GPR()
        GPR.train(X_train, y_train, userkernel=gprsavedkernel, std=y_std, optimizer_restarts=0)
        # Here instead of res, sigma try calculating domain prediction for the test data.

        gpr_pred, GPR_errors = GPR.predict(twoTest, True)
        rf_pred, RF_errors = RF.predict(twoTest, True)
        RF_errors = rfslope * RF_errors + rfintercept

        # Start measuring on different thresholds
        for i_rf_thresholds in range(0, len(rf_thresholds_range)):
            for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
                gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
                rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
                in_domain = accumulator[(rf_thresh, gpr_thresh, 1)]
                out_domain = accumulator[(rf_thresh, gpr_thresh, 0)]
                predictions = [th.predictdomain(GPR_errors[i], RF_errors[i],
                                                gpr_threshold=gpr_thresh, rf_threshold=rf_thresh)
                               for i in range(0, len(twoTest))]

                for i in range(0, len(twoTest)):
                    residual_by_std = (rf_pred[i] - yFrames.to_numpy(dtype=float)[i]) / y_std
                    predicted_error = RF_errors[i]
                    if predictions[i] is 1:
                        in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
                    else:
                        out_domain[th.getcontribution(GPR_errors[i], RF_errors[i],
                                                      gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1]. \
                            append(residual_by_std / predicted_error if predicted_error else 0)

        count += 1
        print(str(count) + ": " + str(testGroup[0]) + "+" + str(testGroup2[0]))
    alreadyDone.append(testGroup[0])
# # Can save these variables so no need to run it again
# with open('LOG_Pair_values.pkl', 'wb') as f:
#     pickle.dump([in_domain, out_domain], f)

in_domain_norm_scores = {a: {b_i: [] for b_i in bin_sizes} for a in normalityTests}
out_domain_norm_scores = {a: {b_i: [] for b_i in bin_sizes} for a in normalityTests}
in_domain_num_points = []
out_domain_num_points = []

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
        in_domain_num_points.append(num_in_domain)
        out_domain_num_points.append(num_out_domain)

        if num_in_domain is 0:
            print('GPR Threshold = {} RF Threshold = {}, No points in-domain'.format(gpr_thresh, rf_thresh))
        score = th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                         _xlabel='RF residual / RF predicted error',
                                         _ylabel='Normalized Counts',
                                         _title='LOG Pair: In-domain RF: {} GPR: {}'.format(rf_thresh,
                                                                                            gpr_thresh),
                                         filename=path + "Plots/",
                                         _bincount=bin_sizes, _normalitytest=normalityTests, _showhist=False
                                         # , temp_file='In_domain_RF-{}_GPR-{}'.format(rf_thresh,
                                         #                                             gpr_thresh)
                                         )
        for test in normalityTests:
            for b_i in bin_sizes:
                in_domain_norm_scores[test][b_i].append(score[test][b_i])

        if num_out_domain is 0:
            print('GPR Threshold = {} RF Threshold = {}, No points out-domain'.format(gpr_thresh, rf_thresh))
        score = th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                         _xlabel='RF residual / RF predicted error',
                                         _ylabel='Normalized Counts',
                                         _title='LOG Pair: Out-domain RF: {} GPR: {}'.format(rf_thresh,
                                                                                             gpr_thresh),
                                         filename=path + "Plots/",
                                         _bincount=bin_sizes, _normalitytest=normalityTests, _showhist=False,
                                         # temp_file='Out_domain_RF-{}_GPR-{}'.format(rf_thresh,
                                         #                                            gpr_thresh)
                                         )
        for test in normalityTests:
            for b_i in bin_sizes:
                out_domain_norm_scores[test][b_i].append(score[test][b_i])

        for test in normalityTests:
            for b_i in bin_sizes:
                cur_result.append(in_domain_norm_scores[test][b_i][-1])
                cur_result.append(out_domain_norm_scores[test][b_i][-1])
        results.append(cur_result)

# For infintity threshold
# Hack to include INF in contour plots
cf_xticks = gpr_thresholds_range
cf_yticks = rf_thresholds_range
if include_INF:
    replace_INF_val_gpr = gpr_thresholds_range[-2] + (gpr_thresholds_range[1] - gpr_thresholds_range[0])
    replace_INF_val_rf = rf_thresholds_range[-2] + (rf_thresholds_range[1] - rf_thresholds_range[0])
    gpr_thresholds_range[-1] = replace_INF_val_gpr
    rf_thresholds_range[-1] = replace_INF_val_rf
    gpr_thresholds[gpr_thresholds == INF] = replace_INF_val_gpr
    rf_thresholds[rf_thresholds == INF] = replace_INF_val_rf
    cf_xticks = np.append(cf_xticks[:-1], 'INF')
    cf_yticks = np.append(cf_yticks[:-1], 'INF')

if make_counts_plot:
    in_domain_num_points = np.array(in_domain_num_points).reshape(
        (len(rf_thresholds_range), len(gpr_thresholds_range)))
    out_domain_num_points = np.array(out_domain_num_points).reshape(
        (len(rf_thresholds_range), len(gpr_thresholds_range)))
    plt.contourf(gpr_thresholds, rf_thresholds, in_domain_num_points)
    plt.colorbar()
    plt.title('LOG Pair Diffusion In-Domain Num Points')
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.xticks(gpr_thresholds_range, cf_xticks)
    plt.yticks(rf_thresholds_range, cf_yticks)
    plt.savefig(path + 'LOG Pair Diffusion In-Domain Num Points.png')
    plt.clf()
    plt.contourf(gpr_thresholds, rf_thresholds, out_domain_num_points)
    plt.colorbar()
    plt.title('LOG Pair Diffusion Out-Domain Num Points')
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.xticks(gpr_thresholds_range, cf_xticks)
    plt.yticks(rf_thresholds_range, cf_yticks)
    plt.savefig(path + 'LOG Pair Diffusion Out-Domain Num Points.png')
    plt.clf()

for test in normalityTests:
    for b_i in bin_sizes:
        in_domain_norm_score_cur = np.array(in_domain_norm_scores[test][b_i]).reshape(
            (len(rf_thresholds_range), len(gpr_thresholds_range)))
        out_domain_norm_score_cur = np.array(out_domain_norm_scores[test][b_i]).reshape(
            (len(rf_thresholds_range), len(gpr_thresholds_range)))
        if contour_plot_same_scale:
            clevels = np.linspace(min(np.min(in_domain_norm_score_cur), np.min(out_domain_norm_score_cur)),
                                  max(np.max(in_domain_norm_score_cur), np.max(out_domain_norm_score_cur)),
                                  10)
        else:
            clevels = None
        plt.contourf(gpr_thresholds, rf_thresholds, in_domain_norm_score_cur, levels=clevels)
        plt.colorbar()
        plt.title('LOG Pair Diffusion In-Domain {} {} bins'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.xticks(gpr_thresholds_range, cf_xticks)
        plt.yticks(rf_thresholds_range, cf_yticks)
        plt.savefig(path + 'LOG Pair Diffusion In-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

        plt.contourf(gpr_thresholds, rf_thresholds, out_domain_norm_score_cur, levels=clevels)
        plt.colorbar()
        plt.title('LOG Pair Diffusion Out-Domain {} {} bins'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.xticks(gpr_thresholds_range, cf_xticks)
        plt.yticks(rf_thresholds_range, cf_yticks)
        plt.savefig(path + 'LOG Pair Diffusion Out-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

fd = open('LOG_Pair_Normality_RMSE_Diffusion.txt', 'w')
log_headers = ["RF cutoff",
               "GPR cutoff",
               "Points in-domain",
               "Points out-domain"]
value_format = [".1f", ".1f", ".0f", ".0f"]

for testname in normalityTests:
    for b_i in bin_sizes:
        log_headers.append('In-Domain {} test score'.format(testname))
        log_headers.append('Out-Domain {} test score'.format(testname))
        value_format.append(".5f")
        value_format.append(".5f")

print(tabulate(results,
               headers=log_headers,
               tablefmt="github",
               floatfmt=value_format), file=fd)
