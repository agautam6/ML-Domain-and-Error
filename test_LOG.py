# generate plots for Diffusion dataset - LOG for RF
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from package import io, rf
from package import testhelper as th

data = io.importdata('data/Diffusion_Data_allfeatures.csv')

groups = data['Material compositions 1'].values

data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

rf_res = np.asarray([])
rf_sigma = np.asarray([])

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()

for train_index, test_index in rfk.split(X, Y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    RF = rf.RF()
    RF.train(X_train, y_train, std=0.4738)
    res, sigma = RF.getrfmetrics(X_test, y_test)
    rf_res = np.concatenate((rf_res, res), axis=None)
    rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)
    print('Hey')

th.RF_plot(rf_res, rf_sigma, "RF", 20)
