import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)


def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data

def getdata(data):
    data = data.drop([24,25,26], axis=1)
    data = data.drop([0])
    return data

def getgpr(X, y):
    kernel = ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    print(gp.kernel_)
    print(gp.log_marginal_likelihood(gp.kernel_.theta))
    return gp

def main():
    data = importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = getdata(data)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.3)
    gpr = getgpr(X_train, y_train)
    # y_pred, sigma = gpr.predict(X_test, return_std=True)
    print(gpr.score(X_test, y_test))

if __name__ == "__main__":
    main()