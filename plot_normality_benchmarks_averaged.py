from scipy import stats
from package import testhelper as th
from package import io
import matplotlib.pyplot as plt
import numpy as np

num_points = 6001
num_runs = 20
bin_sizes = [10, 50, 100, 200, 500]
accumulator = {b_i: np.zeros([num_points]) for b_i in bin_sizes}

for i in range(num_runs):
    print("Run: {}".format(i+1))
    all_points = stats.norm.rvs(loc=0, scale=1, size=num_points, random_state=i)
    arr = []
    for j in range(1, num_points+1):
        print("\tRun: {}, Points: {}".format(i+1, j))
        arr.append(all_points[j-1])
        for b_i in bin_sizes:
            print("\t\tRun: {}, Points: {}, Bin Size: {}".format(i+1, j, b_i))
            n, bins, patches = plt.hist(arr, density=True, bins=b_i, range=(-5, 5))
            score = th.getRMSnormalityscore(n, bins)
            accumulator[b_i][j-1] += score
            plt.clf()

for b_i in bin_sizes:
    accumulator[b_i] = accumulator[b_i] / num_runs
for b_i in bin_sizes:
    plt.plot(list(range(1, num_points+1)), accumulator[b_i])
    plt.title('Normality RMSE Benchmark {} bins'.format(b_i))
    plt.xlabel('No. of samples')
    plt.ylabel('Score')
    plt.savefig('Normality-RMSE-Benchmark-{}-bins.png'.format(b_i))
    plt.clf()
plt.title('Normality RMSE Benchmark')
plt.xlabel('No. of samples')
plt.ylabel('Score')
for b_i in bin_sizes:
    plt.plot(list(range(1, num_points+1)), accumulator[b_i])
plt.legend(bin_sizes, title='Bins')
plt.savefig('Normality-RMSE-Benchmark.png')
plt.clf()
io.savemodelobj(accumulator, filename_from_user='normality_benchmark_rmse_averaged')

for b_i in bin_sizes:
    accumulator[b_i] = -np.log(accumulator[b_i])
for b_i in bin_sizes:
    plt.plot(list(range(1, num_points+1)), accumulator[b_i])
    plt.title('Normality negative log RMSE Benchmark {} bins'.format(b_i))
    plt.xlabel('No. of samples')
    plt.ylabel('Score')
    plt.savefig('Normality-negative-log-RMSE-Benchmark-{}-bins.png'.format(b_i))
    plt.clf()
plt.title('Normality negative log RMSE Benchmark')
plt.xlabel('No. of samples')
plt.ylabel('Score')
for b_i in bin_sizes:
    plt.plot(list(range(1, num_points+1)), accumulator[b_i])
plt.legend(bin_sizes, title='Bins')
plt.savefig('Normality-negative-log-RMSE-Benchmark.png')
plt.clf()
io.savemodelobj(accumulator, filename_from_user='normality_benchmark_negative_log_rmse_averaged')
