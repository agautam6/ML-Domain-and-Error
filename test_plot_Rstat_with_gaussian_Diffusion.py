import statistics
from package import gpr, io, rf, testhelper as th
from sklearn.model_selection import ShuffleSplit

it = 20
snapshots = [i for i in range(1, 21)]
randomstate = 0

trainfile = 'data/Diffusion_Data_allfeatures.csv'
rfslope = 0.65
gprsavedkernel = io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12') \
    .getGPRkernel()

data = io.importdata(trainfile)
data = io.sanitizedata(data)
X_all = data.iloc[:, 1:]
y_all = data.iloc[:, 0]
y_std = statistics.stdev(y_all.to_numpy(dtype=float))

in_domain = []
out_domain = [[], [], []]
count = 1
rs = ShuffleSplit(n_splits=it, test_size=.2, random_state=randomstate)
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
    RF_errors = rfslope * RF_errors
    predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, len(X_test))]
    for i in range(0, len(X_test)):
        residual_by_std = (rf_pred[i] - y_test.to_numpy(dtype=float)[i]) / y_std
        predicted_error = RF_errors[i]
        if predictions[i] is 1:
            in_domain.append(residual_by_std / predicted_error if predicted_error else 0)
        else:
            out_domain[th.getcontribution(GPR_errors[i], RF_errors[i]) - 1].append(
                residual_by_std / predicted_error if predicted_error else 0)
    if count in snapshots:
        if len(in_domain) is not 0:
            th.plotrstatwithgaussian(in_domain, _label=['GPR and RF'],
                                     _xlabel='RF residual / RF predicted error',
                                     _ylabel='Normalized Counts', _title='in-domain Diffusion data {}-iterations'.format(count),
                                     filename='in_domain_Rstat_Diffusion_{}iterations'.format(count))
        else:
            print('{} iterations, No points in-domain'.format(count))
        if len(out_domain[0]) + len(out_domain[1]) + len(out_domain[2]) is not 0:
            th.plotrstatwithgaussian(out_domain, _label=['GPR', 'RF', 'both'],
                                     _xlabel='RF residual / RF predicted error',
                                     _ylabel='Normalized Counts', _title='out-domain Diffusion data {}-iterations'.format(count),
                                     filename='out_domain_Rstat_Diffusion_{}iterations'.format(count))
        else:
            print('{} iterations, No points out-domain'.format(count))
    count += 1
