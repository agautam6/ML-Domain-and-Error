import statistics
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

trainfile = 'data/Diffusion_Data_allfeatures.csv'
rfslope = 0.65
gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12')\
    .getGPRkernel()

data = io.importdata(trainfile)
data = io.sanitizedata(data)
X_all = data.iloc[:, 1:]
y_all = data.iloc[:, 0]
y_std = statistics.stdev(y_all.to_numpy(dtype=float))

in_domain = []
out_domain = []
count = 0
rs = ShuffleSplit(n_splits=20, test_size=.2, random_state=0)
for train_index, test_index in rs.split(X_all):
    print(count)
    count += 1
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train, userkernel=gprsavedkernel, std=y_std, optimizer_restarts=0)
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
