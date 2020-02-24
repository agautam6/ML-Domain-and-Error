# generate plots for PV dataset - both GPR and RF with 1 iteration of random 5-fold CV
import numpy as np
import statistics
from sklearn.model_selection import RepeatedKFold
from package import gpr, io, rf
from package import testhelper as th

data = io.importdata('data/PVstability_Weipaper_alldata_featureselected.csv')
data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])
X_CV = data.iloc[:, 1:]
Y_CV = data.iloc[:, 0]
y_std = statistics.stdev(Y_CV.to_numpy(dtype=float))
rf_res = np.asarray([])
rf_sigma = np.asarray([])
gpr_res = np.asarray([])
gpr_sigma = np.asarray([])
rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=2652124)

gprsavedkernel = io.loadmodelobj('models/GPR_data_PVstability_Weipaper_alldata_featureselected_csv_02-18-20_22-26-49')\
    .getGPRkernel()
ctr = 0
for train_index, test_index in rkf.split(data):
    print("GPR: {}".format(ctr))
    ctr += 1
    X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train, std=y_std, userkernel=gprsavedkernel, optimizer_restarts=0)
    res, sigma = GPR.getgprmetrics(X_test, y_test)
    gpr_res = np.concatenate((gpr_res, res), axis=None)
    gpr_sigma = np.concatenate((gpr_sigma, sigma), axis=None)
th.GPR_plot(gpr_res, gpr_sigma, "GPR", 20)

ctr = 0
for train_index, test_index in rkf.split(data):
    print("RF: {}".format(ctr))
    ctr += 1
    X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
    RF = rf.RF()
    RF.train(X_train, y_train, std=y_std)
    res, sigma = RF.getrfmetrics(X_test, y_test)
    rf_res = np.concatenate((rf_res, res), axis=None)
    rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)
th.RF_plot(rf_res, rf_sigma, "RF", 20)
