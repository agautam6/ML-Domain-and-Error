from sklearn.model_selection import train_test_split, RepeatedKFold
from package import gpr, io
import random

data = io.importdata('data/PVstability_Weipaper_alldata_featureselected.csv')
# data = data.iloc[random.sample(range(0, len(data)), 1000)]
data = io.sanitizedata(data, user_list=['is_testdata', 'Material Composition'])

# 70-30 split
# X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.3)
# GPR_mat = gpr.GPR()
# GPR_mat.train(X_train, y_train)
# GPR_mat.printgprinfo(X_test, y_test)

# GPR_rbf = gpr.GPR()
# GPR_rbf.train(X_train, y_train, kernelchoice=1)
# GPR_rbf.printgprinfo(X_test, y_test)

# Cross validation
X_CV = data.iloc[:, 1:]
Y_CV = data.iloc[:, 0]
rkf = RepeatedKFold(n_splits=5, n_repeats=1)
for train_index, test_index in rkf.split(data):
    X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
    GPR = gpr.GPR()
    GPR.train(X_train, y_train)
    GPR.printgprinfo(X_test, y_test)
    GPR_rbf = gpr.GPR()
    GPR_rbf.train(X_train, y_train, kernelchoice=1)
    GPR_rbf.printgprinfo(X_test, y_test)
