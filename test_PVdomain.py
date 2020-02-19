import statistics
from tabulate import tabulate
from package import gpr, io, rf, testhelper as th

gpr_savedmodel = 'models/GPR_data_PVstability_Weipaper_alldata_featureselected_csv_02-18-20_22-26-49'
rf_savedmodel = 'models/RF_data_PVstability_Weipaper_alldata_featureselected_csv_02-18-20_22-26-54'
# gpr_savedmodel = None
# rf_savedmodel = None
rfslope = 0.89066261
trainfile = 'data/PVstability_Weipaper_alldata_featureselected.csv'
testfile = 'data/PVstability_Weipaper_testdata_featureselected.csv'

data = io.importdata(trainfile)
data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])
X_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0]
y_std = statistics.stdev(y_train.to_numpy(dtype=float))

test_data = io.importdata(testfile)
mcomp_list = test_data['Material Composition'].values
gtype_list = test_data['group_type'].values
test_data = io.sanitizedata(test_data, user_list=['Material Composition', 'group_type',
                                                  'EnergyAboveHull (meV/atom)',
                                                  'Predicted EnergyAboveHull (meV/atom)'])

if gpr_savedmodel is None:
    GPR = gpr.GPR()
    GPR.train(X_train, y_train, std=y_std)
    io.savemodelobj(GPR, 'GPR_' + trainfile)
else:
    GPR = io.loadmodelobj(gpr_savedmodel)

if rf_savedmodel is None:
    RF = rf.RF()
    RF.train(X_train, y_train, std=y_std)
    io.savemodelobj(GPR, 'RF_' + trainfile)
else:
    RF = io.loadmodelobj(rf_savedmodel)

gpr_pred, GPR_errors = GPR.predict(test_data, True)
rf_pred, RF_errors = RF.predict(test_data, True)
RF_errors = rfslope * RF_errors
predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, len(test_data))]
results = [(mcomp_list[i], gtype_list[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in range(0, len(test_data))]

fd = open('PV_domain_results.txt', 'w')
print(tabulate(results, headers=["Material", "Group", "In domain?", "GPR predicted error",
                                 str(rfslope) + " * RF predicted error"]), file=fd)
fd.close()
