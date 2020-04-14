import statistics
import numpy as np
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
import matplotlib.pyplot as plt
from tabulate import tabulate

# Switches
it = 10
randomstate = None
gpr_thresholds_range = np.round(np.arange(0.5, 1.2, 0.1), 1)
rf_thresholds_range = np.round(np.arange(0.5, 1.2, 0.1), 1)
include_INF = True
# normalityTests = ['RMSE', 'Normalized-RMSE', 'Shapiro-Wilk', 'DAgostino-Pearson', 'Log-RMSE', 'Normalized-Log-RMSE']
normalityTests = ['RMSE', 'Normalized-RMSE', 'Log-RMSE', 'Normalized-Log-RMSE']
bin_sizes = [10, 50, 100, 200, 500]
contour_plot_same_scale = False
make_counts_plot = True

# Resources
trainfile = 'data/Diffusion_Data_allfeatures.csv'
rfslope = 0.629705
rfintercept = 0.037300
gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

# Start Test
data = io.importdata(trainfile)
data = io.sanitizedata(data)
X_all = data.iloc[:, 1:]
y_all = data.iloc[:, 0]
y_std = statistics.stdev(y_all.to_numpy(dtype=float))

INF = np.inf
if include_INF:
    gpr_thresholds_range = np.append(gpr_thresholds_range, INF)
    rf_thresholds_range = np.append(rf_thresholds_range, [INF])
gpr_thresholds, rf_thresholds = np.meshgrid(gpr_thresholds_range, rf_thresholds_range)
accumulator = {(r, g, 1): [] for g in gpr_thresholds_range for r in rf_thresholds_range}
accumulator.update({(r, g, 0): [[], [], []] for g in gpr_thresholds_range for r in rf_thresholds_range})

count = 1
rs = ShuffleSplit(n_splits=it, test_size=.3, random_state=randomstate)
for train_index, test_index in rs.split(X_all):
    print(count)
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train, userkernel=gprsavedkernel, std=y_std, optimizer_restarts=0)
    RF = rf.RF()
    RF.train(X_train, y_train, std=y_std)
    gpr_pred, GPR_errors = GPR.predict(X_test, True)
    rf_pred, RF_errors = RF.predict(X_test, True)
    RF_errors = (rfslope * RF_errors) + rfintercept
    for i_rf_thresholds in range(0, len(rf_thresholds_range)):
        for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
            gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
            rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
            in_domain = accumulator[(rf_thresh, gpr_thresh, 1)]
            out_domain = accumulator[(rf_thresh, gpr_thresh, 0)]
            predictions = [th.predictdomain(GPR_errors[i], RF_errors[i],
                                            gpr_threshold=gpr_thresh, rf_threshold=rf_thresh)
                           for i in range(0, len(X_test))]
            for i in range(0, len(X_test)):
                residual_by_std = (rf_pred[i] - y_test.to_numpy(dtype=float)[i]) / y_std
                predicted_error = RF_errors[i]
                if predictions[i] is 1:
                    in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
                else:
                    out_domain[th.getcontribution(GPR_errors[i], RF_errors[i],
                                                  gpr_threshold=gpr_thresh, rf_threshold=rf_thresh) - 1]. \
                        append(residual_by_std / predicted_error if predicted_error else 0)
    count += 1

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
        in_domain_num_points.append(num_in_domain)
        cur_result.append(num_out_domain)
        out_domain_num_points.append(num_out_domain)
        if num_in_domain is 0:
            print('GPR Threshold = {} RF Threshold = {}, No points in-domain'.format(gpr_thresh, rf_thresh))
        score = th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                         _xlabel='RF residual / RF predicted error',
                                         _ylabel='Normalized Counts',
                                         _title='in-domain Diffusion data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                 rf_thresh),
                                         filename='in_domain_Rstat_Diffusion_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                  rf_thresh),
                                         _bincount=bin_sizes, _normalitytest=normalityTests, _showhist=False)
        for test in normalityTests:
            for b_i in bin_sizes:
                in_domain_norm_scores[test][b_i].append(score[test][b_i])
        if num_out_domain is 0:
            print('GPR Threshold = {} RF Threshold = {}, No points out-domain'.format(gpr_thresh, rf_thresh))
        score = th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                         _xlabel='RF residual / RF predicted error',
                                         _ylabel='Normalized Counts',
                                         _title='out-domain Diffusion data GPR: {} RF: {}'.format(gpr_thresh,
                                                                                                  rf_thresh),
                                         filename='out_domain_Rstat_Diffusion_{}-gpr_{}-rf'.format(gpr_thresh,
                                                                                                   rf_thresh),
                                         _bincount=bin_sizes, _normalitytest=normalityTests, _showhist=False)
        for test in normalityTests:
            for b_i in bin_sizes:
                out_domain_norm_scores[test][b_i].append(score[test][b_i])
        for test in normalityTests:
            for b_i in bin_sizes:
                cur_result.append(in_domain_norm_scores[test][b_i][-1])
                cur_result.append(out_domain_norm_scores[test][b_i][-1])
        results.append(cur_result)

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
    plt.title('Diffusion In-Domain Num Points')
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.xticks(gpr_thresholds_range, cf_xticks)
    plt.yticks(rf_thresholds_range, cf_yticks)
    plt.savefig('Diffusion In-Domain Num Points.png')
    plt.clf()
    plt.contourf(gpr_thresholds, rf_thresholds, out_domain_num_points)
    plt.colorbar()
    plt.title('Diffusion Out-Domain Num Points')
    plt.xlabel('GPR cutoff')
    plt.ylabel('RF cutoff')
    plt.xticks(gpr_thresholds_range, cf_xticks)
    plt.yticks(rf_thresholds_range, cf_yticks)
    plt.savefig('Diffusion Out-Domain Num Points.png')
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
        plt.title('Diffusion In-Domain {} {} bins'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.xticks(gpr_thresholds_range, cf_xticks)
        plt.yticks(rf_thresholds_range, cf_yticks)
        plt.savefig('Diffusion In-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

        plt.contourf(gpr_thresholds, rf_thresholds, out_domain_norm_score_cur, levels=clevels)
        plt.colorbar()
        plt.title('Diffusion Out-Domain {} {} bins'.format(test, b_i))
        plt.xlabel('GPR cutoff')
        plt.ylabel('RF cutoff')
        plt.xticks(gpr_thresholds_range, cf_xticks)
        plt.yticks(rf_thresholds_range, cf_yticks)
        plt.savefig('Diffusion Out-Domain {} {} bins.png'.format(test, b_i))
        plt.clf()

fd = open('Normality_tests_Diffusion_logs.txt', 'w')
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
