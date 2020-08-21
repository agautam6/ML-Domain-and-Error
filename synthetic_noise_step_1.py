# This script generates an array of synthetic training data, trains rf and gpr models
# on it, and saves both the data and the models.

import statistics
import numpy as np
from numpy import arange, meshgrid, array, round
from sklearn.model_selection import ShuffleSplit
from package import gpr, io, rf, testhelper as th
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

# Define training data size
training_num = 1000
# Define noise scale factor
scale = 2.0

# Define distribution of x-values:
x0_train=np.random.rand(training_num)*0.5
x1_train=np.random.rand(training_num)*0.5
x2_train=np.random.rand(training_num)*0.5
x3_train=np.random.rand(training_num)*0.5
x4_train=np.random.rand(training_num)*0.5

# Put x-values together in a single array
X_train = [[x0_train[i], x1_train[i], x2_train[i], x3_train[i], x4_train[i]] for i in range(0,training_num)]

# Define y-values with synthetic function
y_train = 30*np.sin(4*np.pi*x0_train*x1_train) + 20*(x2_train - 0.5)**2 + 10*x3_train + 5*x4_train

# Define standard deviation of training data
standard_deviation = np.std(y_train)

#generate noise
mu = 0
sigma = standard_deviation * scale
y_noise = np.random.normal(mu, sigma, len(y_train))

print(np.mean(y_noise))
print(np.std(y_noise))
print(np.std(y_train))

# add noise to y-values
y_train = y_train + y_noise

# Train GPR
GPR = gpr.GPR()
GPR.train_synth(X_train, y_train, std=standard_deviation, kernelchoice=1, optimizer_restarts=30)

# Train RF
RF = rf.RF()
RF.train_synth(X_train, y_train, std=standard_deviation)

# Save models
rf_filename = 'Full_Method_Synthetic_noise/trained_models/whole_training_set_rf_model.sav'
gpr_filename = 'Full_Method_Synthetic_noise/trained_models/whole_training_set_gpr_model.sav'
pickle.dump(GPR, open(gpr_filename, 'wb'))
pickle.dump(RF, open(rf_filename, 'wb'))

# Save training data as np arrays
np.save('Full_Method_Synthetic_noise/training_data/training_x_values.npy', X_train)
np.save('Full_Method_Synthetic_noise/training_data/training_y_values.npy', y_train)


############# Define hypercube test set ##############

# define test data size
test_num = 10000

# x-values: add or remove the *0.5 for each one
x0_test=np.random.rand(test_num)*0.5
x1_test=np.random.rand(test_num)*0.5
x2_test=np.random.rand(test_num)*0.5
x3_test=np.random.rand(test_num)*0.5
x4_test=np.random.rand(test_num)*0.5


X_test = [[x0_test[i], x1_test[i], x2_test[i], x3_test[i], x4_test[i]] for i in range(0,test_num)]

# y-value with friedman function
y_test = 30*np.sin(4*np.pi*x0_test*x1_test) + 20*(x2_test - 0.5)**2 + 10*x3_test + 5*x4_test

# add noise
y_noise_test = np.random.normal(mu, sigma, len(y_test))
y_test = y_test + y_noise_test

# save test data as np arrays
np.save('Full_Method_Synthetic_noise/test_data/test_x_values_hypercube.npy', X_test)
np.save('Full_Method_Synthetic_noise/test_data/test_y_values_hypercube.npy', y_test)

############# Define hypershape test set ##############

# define test data size
test_num = 10000

# x-values: add or remove the *0.5 for each one
x0_test=np.random.rand(test_num)
x1_test=np.random.rand(test_num)*0.5
x2_test=np.random.rand(test_num)*0.5
x3_test=np.random.rand(test_num)*0.5
x4_test=np.random.rand(test_num)*0.5


X_test = [[x0_test[i], x1_test[i], x2_test[i], x3_test[i], x4_test[i]] for i in range(0,test_num)]

# y-value with friedman function
y_test = 30*np.sin(4*np.pi*x0_test*x1_test) + 20*(x2_test - 0.5)**2 + 10*x3_test + 5*x4_test

# add noise
y_noise_test = np.random.normal(mu, sigma, len(y_test))
y_test = y_test + y_noise_test

# save test data as np arrays
np.save('Full_Method_Synthetic_noise/test_data/test_x_values_hypershape.npy', X_test)
np.save('Full_Method_Synthetic_noise/test_data/test_y_values_hypershape.npy', y_test)

