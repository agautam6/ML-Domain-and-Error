# generate plots for Diffusion dataset - LOG Pair for RF
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from package import io, rf, testhelper as th


def checkAlreadyDone(element, alreadylist):
    for x in alreadylist:
        if element == x:
            return True
    return False


filename = os.path.abspath('.') + "/Plots/slope-intercept/Diffusion LOG Pair/LOGPair"
data = io.importdata('data/Diffusion_Data_allfeatures.csv')
groups = data['Material compositions 1'].values

data = io.sanitizedata(data)
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
rf_res = np.asarray([])
rf_sigma = np.asarray([])

# Leave out group test for Material compositions 1
rfk = LeaveOneGroupOut()
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
        res, sigma = RF.getrfmetrics(X_test, y_test)
        rf_res = np.concatenate((rf_res, res), axis=None)
        rf_sigma = np.concatenate((rf_sigma, sigma), axis=None)

    alreadyDone.append(testGroup[0])

th.RF_plot(rf_res, rf_sigma, "RF", 20,
           filename)
