import statistics

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from package import gpr, io, rf, testhelper as th

data = io.importdata('data/Diffusion_Data_allfeatures.csv')
groups = data['Material compositions 1'].values

data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
rf_res = np.asarray([])
rf_sigma = np.asarray([])
rfslope = 1.143641
y_std = statistics.stdev(Y.to_numpy(dtype=float))

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
j = 0
first = 1
startIndex = 1
in_domain = []
out_domain = [[], [], []]

gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

for train_index, test_index in rfk.split(X, Y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    RF = rf.RF()
    RF.train(X_train, y_train, std=0.4738)

    GPR = gpr.GPR()
    GPR.train(X_train, y_train, userkernel=gprsavedkernel, std=y_std, optimizer_restarts=0)
    # Here instead of res, sigma try calculating domain prediction for the test data.

    gpr_pred, GPR_errors = GPR.predict(X_test, True)
    rf_pred, RF_errors = RF.predict(X_test, True)
    RF_errors = rfslope * RF_errors

    predictions = [th.predictdomainWithThreshold(GPR_errors[i], RF_errors[i], 0.8) for i in range(0, len(X_test))]
    results = [(groups[j + i], predictions[i], GPR_errors[i], RF_errors[i]) for i in
               range(0, len(X_test))]
    j = j + len(X_test)

    for i in range(0, len(X_test)):
        residual_by_std = (rf_pred[i] - y_test.to_numpy(dtype=float)[i]) / y_std
        predicted_error = RF_errors[i]
        if predictions[i] is 1:
            in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
        else:
            out_domain[th.getcontribution(GPR_errors[i], RF_errors[i]) - 1].append(
                residual_by_std / predicted_error if predicted_error else 0)

    testGroup = np.delete(groups, train_index)
    element = testGroup[0]

    if len(in_domain) is not 0:
        th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                 _xlabel='RF residual / RF predicted error',
                                 _ylabel='Normalized Counts', _title='LOG: In-domain Test On {}'.format(element),
                                 filename='in_domain_Rstat_Diffusion_LOG_{}'.format(element), _bincount=30)
    else:
        print('{} iterations, No points in-domain'.format(element))
    if len(out_domain[0]) + len(out_domain[1]) + len(out_domain[2]) is not 0:
        th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                 _xlabel='RF residual / RF predicted error',
                                 _ylabel='Normalized Counts',
                                 _title='LOG: Out-domain Test On {}'.format(element),
                                 filename='out_domain_Rstat_Diffusion_LOG_{}'.format(element), _bincount=30)
    else:
        print('{} iterations, No points out-domain'.format(element))
