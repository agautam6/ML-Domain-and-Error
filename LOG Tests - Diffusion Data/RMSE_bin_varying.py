import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tabulate import tabulate

from package import gpr, io, rf, testhelper as th


def checkAlreadyDone(element, alreadylist):
    for x in alreadylist:
        if element == x:
            return True
    return False


# Data Collection
data = io.importdata('../data/Diffusion_Data_allfeatures.csv')
# data = io.importdata('../data/temp.csv')

groups = data['Material compositions 1'].values
data = io.sanitizedata(data)
gprsavedkernel = io.loadmodelobj('../models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
rfslope = 0.919216
rfintercept = -0.025370
y_std = statistics.stdev(Y.to_numpy(dtype=float))

# Setup thresholds

gpr_thresholds_range = round(np.arange(0.5, 1.2, 0.1), 1)
rf_thresholds_range = round(np.arange(0.5, 1.2, 0.1), 1)
normalityTests = ['RMSE']
defaults = {'RMSE': 1, 'Shapiro-Wilk': 0, 'DAgostino-Pearson': 0}
bin_sizes = [10, 50, 100, 200, 500]
gpr_thresholds, rf_thresholds = np.meshgrid(gpr_thresholds_range, rf_thresholds_range)
accumulator = {(r, g, 1): [] for g in gpr_thresholds_range for r in rf_thresholds_range}
accumulator.update({(r, g, 0): [[], [], []] for g in gpr_thresholds_range for r in rf_thresholds_range})

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
j = 0
w = 0
first = 1
in_domain = []
out_domain = [[], [], []]
alreadyDone = []
count = 0

# path = os.path.abspath('..') + "/domain results/Normality Test RMSE/Diffusion LOG/"
path = os.path.abspath('..') + "/domain results/RMSE varying bin-widths/Diffusion LOG Pair/"

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
                # in_domain_curr = []
                # out_domain_curr = [[], [], []]

                predictions = [th.predictdomain(GPR_errors[i], RF_errors[i],
                                                gpr_threshold=gpr_thresh, rf_threshold=rf_thresh)
                               for i in range(0, len(twoTest))]

                for i in range(0, len(twoTest)):
                    residual_by_std = (rf_pred[i] - yFrames.to_numpy(dtype=float)[i]) / y_std
                    predicted_error = RF_errors[i]
                    if predictions[i] is 1:
                        in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
                        # in_domain_curr.append(residual_by_std / predicted_error if predicted_error else 0)
                    else:
                        out_domain[th.getcontribution(GPR_errors[i], RF_errors[i],
                                                      gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1]. \
                            append(residual_by_std / predicted_error if predicted_error else 0)
                        # out_domain_curr[th.getcontribution(GPR_errors[i], RF_errors[i],
                        #                                    gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1]. \
                        #     append(residual_by_std / predicted_error if predicted_error else 0)

                # Print the plots of all the metals along with thresholds.

                # element = str(testGroup[0]) + "+" + str(testGroup2[0])
                # currFolder = path
                # + "RF_Threshold=" + str(rf_thresh) + ",GPR_Threshold=" + str(gpr_thresh) + "/"

                # if len(in_domain_curr) is not 0:
                #     th.plotrstatwithgaussian(in_domain_curr, _label=['GPR and RF'],
                #                              _xlabel='RF residual / RF predicted error',
                #                              _ylabel='Normalized Counts',
                #                              _title='LOG Pair: In-domain on {} for GPR: {} RF: {}'.format(element,
                #                                                                                           gpr_thresh,
                #                                                                                           rf_thresh),
                #                              filename=currFolder + "Plots/",
                #                              _bincount=bin_sizes, _normalitytest=['RMSE'],
                #                              temp_file="RF_Threshold=" + str(rf_thresh) + ",GPR_Threshold=" + str(
                #                                  gpr_thresh) + "/"
                #                                        + 'in_domain_{}'.format(element))
                # else:
                #     print('{} iterations, No points in-domain'.format(element))
                #
                # if len(out_domain_curr[0]) + len(out_domain_curr[1]) + len(out_domain_curr[2]) is not 0:
                #     th.plotrstatwithgaussian(out_domain_curr, _label=['GPR', 'RF', 'both'],
                #                              _xlabel='RF residual / RF predicted error',
                #                              _ylabel='Normalized Counts',
                #                              _title='LOG Pair: Out-domain on {} for GPR: {} RF: {}'.format(element,
                #                                                                                            gpr_thresh,
                #                                                                                            rf_thresh),
                #                              filename=currFolder + "Plots/",
                #                              _bincount=bin_sizes, _normalitytest=['RMSE'],
                #                              temp_file="RF_Threshold=" + str(rf_thresh) + ",GPR_Threshold=" + str(
                #                                  gpr_thresh) + "/"
                #                                        + 'out_domain_{}'.format(element))
                # else:
                #     print('{} iterations, No points out-domain'.format(element))

        count += 1
        print(count)
        print(str(testGroup[0]) + "+" + str(testGroup2[0]))
    alreadyDone.append(testGroup[0])

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
                                             _title='LOG Pair: In-domain RF: {} GPR: {}'.format(rf_thresh,
                                                                                                gpr_thresh),
                                             filename=path + "Plots/",
                                             _bincount=bin_sizes, _normalitytest=normalityTests,
                                             # temp_file='In_domain_RF-{}_GPR-{}'.format(rf_thresh,
                                             #                                           gpr_thresh)
                                             )
            # for test in normalityTests:
            #     for b_i in bin_sizes:
            #         in_domain_norm_scores[test][b_i].append(score[test][b_i])

            for b_i in bin_sizes:
                in_domain_norm_scores['RMSE'][b_i].append(score['RMSE'][b_i])

        else:
            print('GPR Threshold = {} RF Threshold = {}, No points in-domain'.format(gpr_thresh, rf_thresh))
            # for test in normalityTests:
            #     for b_i in bin_sizes:
            #         in_domain_norm_scores[test][b_i].append(score[test][b_i])

            for b_i in bin_sizes:
                in_domain_norm_scores['RMSE'][b_i].append(score['RMSE'][b_i])

        if num_out_domain is not 0:
            score = th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                             _xlabel='RF residual / RF predicted error',
                                             _ylabel='Normalized Counts',
                                             _title='LOG Pair: Out-domain RF: {} GPR: {}'.format(rf_thresh,
                                                                                                 gpr_thresh),
                                             filename=path + "Plots/",
                                             _bincount=bin_sizes, _normalitytest=normalityTests,
                                             # temp_file='Out_domain_RF-{}_GPR-{}'.format(rf_thresh,
                                             #                                            gpr_thresh)
                                             )
            # for test in normalityTests:
            #     for b_i in bin_sizes:
            #         out_domain_norm_scores[test][b_i].append(score[test][b_i])
            for b_i in bin_sizes:
                out_domain_norm_scores['RMSE'][b_i].append(score['RMSE'][b_i])

        else:
            print('GPR Threshold = {} RF Threshold = {}, No points out-domain'.format(gpr_thresh, rf_thresh))
            # for test in normalityTests:
            #     for b_i in bin_sizes:
            #         out_domain_norm_scores[test][b_i].append(score[test][b_i])
            for b_i in bin_sizes:
                out_domain_norm_scores['RMSE'][b_i].append(score['RMSE'][b_i])

        # for test in normalityTests:
        #     for b_i in bin_sizes:
        #         cur_result.append(in_domain_norm_scores[test][b_i][-1])
        #         cur_result.append(out_domain_norm_scores[test][b_i][-1])

        for b_i in bin_sizes:
            cur_result.append(in_domain_norm_scores['RMSE'][b_i][-1])
            cur_result.append(out_domain_norm_scores['RMSE'][b_i][-1])
        results.append(cur_result)

test = 'RMSE'
#
# for test in normalityTests:
for b_i in bin_sizes:
    in_domain_norm_score_cur = np.array(in_domain_norm_scores[test][b_i]).reshape(
        (len(rf_thresholds_range), len(gpr_thresholds_range)))
    plt.contourf(gpr_thresholds, rf_thresholds, in_domain_norm_score_cur)
    plt.colorbar()
    plt.title('Diffusion LOG In-Domain {} {} bins'.format(test, b_i))
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.savefig(path + 'Diffusion LOG In-Domain {} {} bins.png'.format(test, b_i))
    plt.clf()

    out_domain_norm_score_cur = np.array(out_domain_norm_scores[test][b_i]).reshape(
        (len(rf_thresholds_range), len(gpr_thresholds_range)))
    plt.contourf(gpr_thresholds, rf_thresholds, out_domain_norm_score_cur)
    plt.colorbar()
    plt.title('Diffusion LOG Out-Domain {} {} bins.png'.format(test, b_i))
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.savefig(path + 'Diffusion LOG Out-Domain {} {} bins.png'.format(test, b_i))
    plt.clf()

fd = open('LOG_Normality_RMSE_Diffusion.txt', 'w')
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
