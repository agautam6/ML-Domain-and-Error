import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def plot(res, sigma, stdev):
    ########### REAL RFDT DATA GOES HERE ##############

    # Data for Random Forest model errors
    # RF_model_errors = np.random.random_sample(400, ) * 1.5
    # Data for Random Forest absolute residuals
    # RF_abs_res = np.random.random_sample(400, )
    # Value of Dataset Standard Deviation
    # RF_stdev = 0.4738  # Actual value is 0.4738

    RF_model_errors = sigma
    RF_abs_res = res
    RF_stdev = stdev

    # %%

    # Divide RF data by RF standard deviation
    RF_model_errors = RF_model_errors / RF_stdev
    RF_abs_res = RF_abs_res / RF_stdev

    # %%

    # Create initial scatter plot
    plt.xlabel("RF model errors / dataset stdev")
    plt.ylabel("RF Absolute residuals / dataset stdev")
    plt.title("RFDT Absolute Residuals vs. Model Errors")
    plt.plot(RF_model_errors, RF_abs_res, '.', color='blue');

    # %%

    plt.show()

    # Set number of bins for RMS calculation
    RF_number_of_bins = 20
    # Histogram of RF error bin counts
    plt.hist(RF_model_errors, bins=RF_number_of_bins, color='blue', edgecolor='black')
    plt.xlabel("RF model errors / dataset stdev")
    plt.ylabel("Counts")
    plt.title("RFDT Bin Counts");

    # %%

    plt.show()

    # Set bins for calculating RMS
    RF_upperbound = np.amax(RF_model_errors)
    RF_lowerbound = np.amin(RF_model_errors)
    RF_bins = np.linspace(RF_lowerbound, RF_upperbound, RF_number_of_bins)
    # Create a vector determining bin of each data point
    RF_digitized = np.digitize(RF_model_errors, RF_bins)
    # Calculate RMS of the absolute residuals
    RF_RMS_abs_res = [np.sqrt((RF_abs_res[RF_digitized == i] ** 2).mean()) for i in range(1, len(RF_bins))]

    # %%

    # Set the x-values to the midpoint of each bin
    RF_start = (RF_bins[0] + RF_bins[1]) / 2
    RF_end = RF_bins[len(RF_bins) - 1] - RF_start
    RF_binned_model_errors = np.linspace(RF_start, RF_end, len(RF_bins) - 1)

    # %%

    # Fit a line to the data
    # RF_model = LinearRegression(fit_intercept=False)
    #
    # RF_model.fit(RF_binned_model_errors[:, np.newaxis],
    #              RF_RMS_abs_res)  #### SELF: Can indicate subset of points to fit to using ":" --> "a:b"
    #
    # RF_xfit = np.linspace(0, RF_upperbound, RF_number_of_bins - 1)
    # RF_yfit = RF_model.predict(RF_xfit[:, np.newaxis])

    # %%

    # Create RMS scatter plot
    plt.xlabel("RF model errors / dataset stdev")
    plt.ylabel("RF RMS Absolute residuals / dataset stdev")
    # plt.ylim(0,1)
    plt.title("RFDT RMS Absolute Residuals vs. Model Errors")
    plt.plot(RF_binned_model_errors, RF_RMS_abs_res, 'o', color='blue')
    # plt.plot(RF_xfit, RF_yfit);

    # %%

    plt.show()

    # r2_score(RF_RMS_abs_res, RF_yfit)

    # %%

# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data

# Sanitize data. Temporarily dropping columns E_regression, Material Composition
def getdata(data):
    data = data.drop([24,25,26], axis=1)
    data = data.drop([0])
    return data

def getgprmetrics(X_train, y_train, X_test, y_test):
    kernel = ConstantKernel() + 1.0 ** 2 * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    # print(gp.kernel_)
    # print(gp.log_marginal_likelihood(gp.kernel_.theta))
    # print(gpr.score(X_test, y_test))
    y_pred, sigma = gp.predict(X_test, return_std=True)
    y_test = y_test.to_numpy(dtype=float)
    y_std = statistics.stdev(y_test)
    residual = abs(y_pred - y_test)
    return residual, sigma, y_std

def getrfmetrics(X_train, y_train, X_test, y_test):
    regressor = RandomForestRegressor(n_estimators=15, random_state=8)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_test = y_test.to_numpy(dtype=float)
    y_residual = abs(y_pred - y_test)
    error = []
    for x in range(len(X_test)):
        preds = []
        for pred in regressor.estimators_:
            preds.append(pred.predict([X_test[x]])[0])
        error.append(statistics.stdev(preds))
    error = np.array(error)
    return y_residual, error, statistics.stdev(y_test)

def main():
    data = importdata('_haijinlogfeaturesnobarrier_alldata.csv')
    data = getdata(data)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.3)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    res, sigma, std = getgprmetrics(X_train, y_train, X_test, y_test)
    plot(res, sigma, std)
    res, sigma, std = getrfmetrics(X_train, y_train, X_test, y_test)
    plot(res, sigma, std)

if __name__ == "__main__":
    main()