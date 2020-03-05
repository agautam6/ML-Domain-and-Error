# generate plots for Diffusion dataset - LOG for RF
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tabulate import tabulate

from package import io, rf, gpr
from package import testhelper as th


def checkAlreadyDone(element, alreadylist):
    for x in alreadylist:
        if element == x:
            return True
    return False


data = io.importdata('data/Diffusion_Data_allfeatures.csv')
groups = data['Material compositions 1'].values
gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
rf_res = np.asarray([])
rf_sigma = np.asarray([])
rfslope = 0.927880
y_std = statistics.stdev(Y.to_numpy(dtype=float))

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
j = 0
w = 0
first = 1
fd = open('domain results/LOGPair_DomainPrediction.txt', 'a+')
in_domain = []
out_domain = [[], [], []]

pdtabulate = lambda df: tabulate(df, headers=["Material", "In domain?", "GPR predicted error",
                                              str(rfslope) + " * RF predicted Error"], tablefmt='psql')

alreadyDone = []

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
        RF.train(X_train, y_train, std=0.4738)

        GPR = gpr.GPR()
        GPR.train(X_train, y_train, userkernel=gprsavedkernel, std=y_std, optimizer_restarts=0)
        # Here instead of res, sigma try calculating domain prediction for the test data.

        gpr_pred, GPR_errors = GPR.predict(twoTest, True)
        rf_pred, RF_errors = RF.predict(twoTest, True)
        RF_errors = rfslope * RF_errors

        predictions = [th.predictdomain(GPR_errors[i], RF_errors[i], threshold=0.8) for i in range(0, len(twoTest))]
        results = [(testFinal[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in
                   range(0, len(twoTest))]
        # j = 0

        for i in range(0, len(twoTest)):
            residual_by_std = abs(rf_pred[i] - yFrames.to_numpy(dtype=float)[i]) / y_std
            predicted_error = RF_errors[i]
            if predictions[i] is 1:
                in_domain.append(residual_by_std / predicted_error)
            else:
                out_domain[th.getcontribution(GPR_errors[i], RF_errors[i]) - 1].append(
                    residual_by_std / predicted_error)

        print(pdtabulate(results), file=fd)

    alreadyDone.append(testGroup[0])

fd.close()
plt.hist(in_domain)
plt.ylabel('Counts')
plt.xlabel('RF absolute residual / RF predicted error')
plt.savefig("in_domain_Test.png")
plt.clf()
plt.hist(out_domain, stacked=True, label=['GPR', 'RF', 'both'])
plt.legend(prop={'size': 10})
plt.ylabel('Counts')
plt.xlabel('RF absolute residual / RF predicted error')
plt.savefig("out_domain_Test.png")
plt.clf()
