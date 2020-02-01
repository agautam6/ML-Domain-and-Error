from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
import statistics
from sklearn.preprocessing import StandardScaler


class GPR:
    gp = None
    sc = None
    X_train = None
    y_train = None
    kernel = None

    def __init__(self):
        pass

    def train(self, X_train, y_train):
        # Scale features
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train = y_train
        self.kernel = ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10).fit(self.X_train, self.y_train)

    def predict(self, x_test, retstd=True):
        x_pred = self.sc.fit_transform(x_test)
        return self.gp.predict(x_pred, return_std=retstd)

    def getgprmetrics(self, X_test, y_test):
        X_pred = self.sc.fit_transform(X_test)
        y_pred, sigma = self.gp.predict(X_pred, return_std=True)
        y_test1 = y_test.to_numpy(dtype=float)
        y_std = statistics.stdev(y_test1)
        residual = abs(y_pred - y_test1)
        return residual/y_std, sigma/y_std

    def printgprinfo(self, X_test=None, y_test=None):
        if self.gp is None:
            print('GPR model not trained\n')
            return
        print(self.gp.kernel_)
        print(self.gp.log_marginal_likelihood(self.gp.kernel_.theta))
        if X_test is not None and y_test is not None:
            print(self.gp.score(X_test, y_test))
