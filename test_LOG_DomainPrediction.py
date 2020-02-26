# generate plots for Diffusion dataset - LOG for RF
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from tabulate import tabulate

from package import io, rf, gpr
from package import testhelper as th

data = io.importdata('data/Diffusion_Data_allfeatures.csv')
groups = data['Material compositions 1'].values

data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
rf_res = np.asarray([])
rf_sigma = np.asarray([])
rfslope = 1.143641

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
j = 0
first = 1
fd = open('domain results/Diffusion_DataSet_LOG__domain_results.txt', 'a+')
table = []

pdtabulate = lambda df: tabulate(df, headers=["Material", "In domain?", "GPR predicted error",
                                              str(rfslope) + " * RF predicted Error"], tablefmt='psql')

for train_index, test_index in rfk.split(X, Y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    RF = rf.RF()
    RF.train(X_train, y_train, std=0.4738)

    GPR = gpr.GPR()
    GPR.train(X_train, y_train, std=0.4738, optimizer_restarts=3)
    # Here instead of res, sigma try calculating domain prediction for the test data.

    gpr_pred, GPR_errors = GPR.predict(X_test, True)
    rf_pred, RF_errors = RF.predict(X_test, True)
    RF_errors = rfslope * RF_errors

    predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, len(X_test))]
    results = [(groups[j + i], predictions[i], GPR_errors[i], RF_errors[i]) for i in
               range(0, len(X_test))]
    j = j + len(X_test)

    print(pdtabulate(results), file=fd)
fd.close()
