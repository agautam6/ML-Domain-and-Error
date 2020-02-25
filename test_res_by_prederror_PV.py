import statistics
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

trainfile = 'data/PVstability_Weipaper_alldata_featureselected.csv'
rfslope = 0.89066261
gprsavedkernel = io.loadmodelobj('models/GPR_data_PVstability_Weipaper_alldata_featureselected_csv_02-18-20_22-26-49')\
    .getGPRkernel()

data = io.importdata(trainfile)
data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])
X_all = data.iloc[:, 1:]
y_all = data.iloc[:, 0]
y_std = statistics.stdev(y_all.to_numpy(dtype=float))

in_domain = []
out_domain = []

rs = ShuffleSplit(n_splits=10, test_size=.2, random_state=0)
for train_index, test_index in rs.split(X_all):
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train, std=y_std, userkernel=gprsavedkernel, optimizer_restarts=0)
    RF = rf.RF()
    RF.train(X_train, y_train, std=y_std)
    gpr_pred, GPR_errors = GPR.predict(X_test, True)
    rf_pred, RF_errors = RF.predict(X_test, True)
    RF_errors = rfslope * RF_errors
    predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, len(X_test))]
    for i in range(0, len(X_test)):
        residual_by_std = abs(rf_pred[i]-y_test.to_numpy(dtype=float)[i])/y_std
        predicted_error = RF_errors[i]
        if predictions[i] is 1:
            in_domain.append(residual_by_std/predicted_error)
        else:
            out_domain.append(residual_by_std/predicted_error)

plt.hist(in_domain)
plt.ylabel('Counts')
plt.xlabel('RF absolute residual / RF predicted error')
plt.savefig("in_domain.png")
plt.clf()
plt.hist(out_domain)
plt.ylabel('Counts')
plt.xlabel('RF absolute residual / RF predicted error')
plt.savefig("out_domain.png")
plt.clf()
