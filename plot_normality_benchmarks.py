from scipy import stats
from package import testhelper as th
# from package import io
import matplotlib.pyplot as plt
# normalityTests = ['RMSE', 'Shapiro-Wilk', 'DAgostino-Pearson']
normalityTests = ['RMSE']
bin_sizes = [10, 50, 100, 200, 500]
all_points = stats.norm.rvs(loc=0, scale=1, size=6001)
results = {a: {b_i: [] for b_i in bin_sizes} for a in normalityTests}
count = 10
x_vals = []
while count < len(all_points):
    print(count)
    arr = all_points[:count]
    scores = th.plotrstatwithgaussian(arr, _bincount=bin_sizes, filename='dummy', _normalitytest=normalityTests)
    x_vals.append(count)
    for k in scores:
        for b_i in scores[k]:
            results[k][b_i].append(scores[k][b_i])
    count += 10
for k in normalityTests:
    for b_i in bin_sizes:
        plt.plot(x_vals, results[k][b_i])
        plt.title('Normality Benchmark {} {} bins'.format(k, b_i))
        plt.xlabel('No. of samples')
        plt.ylabel('Score')
        plt.savefig('Normality-Benchmark-{}-test-{}-bins.png'.format(k, b_i))
        # plt.show()
        plt.clf()
# io.savemodelobj(results['RMSE'], filename_from_user='normality_benchmark_rmse')
