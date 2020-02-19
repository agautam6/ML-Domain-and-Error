import pickle
from datetime import datetime as dt
import pytz as tz
import statistics
from tabulate import tabulate
from package import gpr
from package import io
from package import rf
from package import testhelper as th


def getModelSaveFileName(filename):
    return filename.replace("/", "_").replace(".", "_") + "_" \
           + dt.now(tz=tz.timezone('America/Chicago')).strftime("%m-%d-%y_%H-%M-%S")


gpr_savedmodel = None
rf_savedmodel = None
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
    with open('GPR_' + getModelSaveFileName(trainfile) + '.pkl', 'wb') as out:
        pickle.dump(GPR, out, pickle.HIGHEST_PROTOCOL)
else:
    with open(gpr_savedmodel, 'rb') as inp:
        GPR = pickle.load(inp)

if rf_savedmodel is None:
    RF = rf.RF()
    RF.train(X_train, y_train, std=y_std)
    with open('RF_' + getModelSaveFileName(trainfile) + '.pkl', 'wb') as out:
        pickle.dump(RF, out, pickle.HIGHEST_PROTOCOL)
else:
    with open(rf_savedmodel, 'rb') as inp:
        RF = pickle.load(inp)

gpr_pred, GPR_errors = GPR.predict(test_data, True)
rf_pred, RF_errors = RF.predict(test_data, True)
RF_errors = rfslope * RF_errors
predictions = [th.predictdomain(GPR_errors[i], RF_errors[i]) for i in range(0, len(test_data))]
results = [(mcomp_list[i], gtype_list[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in range(0, len(test_data))]

fd = open('PV_domain_results.txt', 'w')
print(tabulate(results, headers=["Material", "Group", "In domain?", "GPR predicted error",
                                 str(rfslope) + " * RF predicted error"]), file=fd)
fd.close()
