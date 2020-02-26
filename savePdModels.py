from package import gpr, io, rf
import statistics
trainfile = 'data/Diffusion_Data_allfeatures.csv'
data = io.importdata(trainfile)
data = io.sanitizedata(data)
X_all = data.iloc[:, 1:]
y_all = data.iloc[:, 0]
data = io.importdata(trainfile)
data = io.sanitizedata(data)
X_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0]
y_std = statistics.stdev(y_train.to_numpy(dtype=float))
GPR = gpr.GPR()
GPR.train(X_train, y_train, std=y_std)
io.savemodelobj(GPR, 'GPR_' + trainfile)
RF = rf.RF()
RF.train(X_train, y_train, std=y_std)
io.savemodelobj(GPR, 'RF_' + trainfile)
