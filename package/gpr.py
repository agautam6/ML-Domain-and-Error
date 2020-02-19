import statistics

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern, ConstantKernel, WhiteKernel, RBF)
from sklearn.preprocessing import StandardScaler


class GPR:
    gp = None
    sc = None
    X_train = None
    y_train = None
    kernel = None
    y_std_train = None

    def __init__(self):
        pass

    # kernelchoice controls what kernel to use from inbuilt kernels:
    # kernelchoice=0: ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
    # kernelchoice=1: ConstantKernel()*RBF()
    #
    # userkernel can be used to provide a custom kernel. The preference is given to 'kernelchoice' over 'userkernel'
    def train(self, X_train, y_train, std=None, kernelchoice=0, userkernel=None, optimizer_restarts=10):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        y_train_temp = self.y_train.to_numpy(dtype=float)
        if std is None:
            self.y_std_train = statistics.stdev(y_train_temp)
        else:
            self.y_std_train = std
        if kernelchoice is 0:
            self.kernel = ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
            self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                               n_restarts_optimizer=optimizer_restarts).fit(self.X_train, self.y_train)
        elif kernelchoice is 1:
            # Ryan's kernel
            self.kernel = ConstantKernel()*RBF()
            self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                               alpha=0.00001,
                                               n_restarts_optimizer=optimizer_restarts,
                                               normalize_y=False).fit(self.X_train, self.y_train)

        elif userkernel is not None:
            # User defined kernel
            self.kernel = userkernel
            self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                               n_restarts_optimizer=optimizer_restarts).fit(self.X_train, self.y_train)
        else:
            raise ValueError('ERROR: Invalid GPR kernel.')

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.transform(x_test)
        pred, std = self.gp.predict(x_pred, return_std=retstd)
        return pred, std / self.y_std_train

    def getgprmetrics(self, X_test, y_test):
        X_pred = self.sc.transform(X_test)
        y_pred, sigma = self.gp.predict(X_pred, return_std=True)
        y_test1 = y_test.to_numpy(dtype=float)
        # y_std = statistics.stdev(y_test1)
        residual = abs(y_pred - y_test1)
        # return residual / y_std, sigma / y_std
        return residual / self.y_std_train, sigma / self.y_std_train

    def printgprinfo(self, X_test=None, y_test=None):
        if self.gp is None:
            print('GPR model not trained\n')
            return
        print(self.gp.kernel_)
        print(self.gp.log_marginal_likelihood(self.gp.kernel_.theta))
        if X_test is not None and y_test is not None:
            print(self.gp.score(self.sc.transform(X_test), y_test))

    def getGPRkernel(self):
        return self.gp.kernel_
