from package import io, gpr
from sklearn.model_selection import RepeatedKFold

# Runs 5-fold cv on "alldata" data set using two GPR kernel variations : RBF and Matern
data = io.importdata('data/_haijinlogfeaturesnobarrier_alldata.csv')
data = io.sanitizedata(data)
X_CV = data.iloc[:, :-1]
Y_CV = data.iloc[:, -1]
rkf = RepeatedKFold(n_splits=5, n_repeats=1)
for train_index, test_index in rkf.split(data):
    X_train, X_test = X_CV.iloc[train_index], X_CV.iloc[test_index]
    y_train, y_test = Y_CV.iloc[train_index], Y_CV.iloc[test_index]
    GPR1 = gpr.GPR()
    GPR1.train(X_train, y_train)
    GPR1.printgprinfo(X_test, y_test)
    GPR2 = gpr.GPR()
    GPR2.train(X_train, y_train, kernelchoice=1)
    GPR2.printgprinfo(X_test, y_test)
