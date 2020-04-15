import statistics
import numpy as np
from numpy import arange, meshgrid, array, round
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

it = 10
randomstate = None
gpr_thresholds_range = round(arange(0.1, 1.2, 0.1), 1)
rf_thresholds_range = round(arange(0.1, 1.2, 0.1), 1)
# normalityTests = ['RMSE', 'Shapiro-Wilk', 'DAgostino-Pearson']
normalityTests = ['RMSE']
defaults = {'RMSE': 1, 'Shapiro-Wilk': 0, 'DAgostino-Pearson': 0}
bin_sizes = [10, 50, 100, 200, 500]

rfslope = 0.256
rfintercept = 0.0184


# define training data size
training_num = 10000

# x-values: all uniformly distributed between 0 and 1
x0_train=np.random.rand(training_num)*0.5
x1_train=np.random.rand(training_num)*0.5
x2_train=np.random.rand(training_num)*0.5
x3_train=np.random.rand(training_num)*0.5
x4_train=np.random.rand(training_num)*0.5

X_train = [[x0_train[i], x1_train[i], x2_train[i], x3_train[i], x4_train[i]] for i in range(0,training_num)]

# y-value with friedman function
y_train = 30*np.sin(4*np.pi*x0_train*x1_train) + 20*(x2_train - 0.5)**2 + 10*x3_train + 5*x4_train

# Define standard deviation of training data
standard_deviation = np.std(y_train)

# Train GPR
GPR = gpr.GPR()
GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=30)

# Train RF
RF = rf.RF()
RF.train_synth(X_train, y_train, std=standard_deviation)

# Save models
rf_filename = 'rf_synth_model_10000.sav'
gpr_filename = 'gpr_synth_model_10000.sav'
pickle.dump(GPR, open(gpr_filename, 'wb'))
pickle.dump(RF, open(rf_filename, 'wb'))