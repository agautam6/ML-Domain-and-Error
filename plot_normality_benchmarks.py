from scipy import stats
from package import testhelper as th
import matplotlib.pyplot as plt
normalityTests = ['RMSE', 'Shapiro-Wilk', 'DAgostino-Pearson']
all_points = stats.norm.rvs(loc=0, scale=1, size=6001)
results = {a: [] for a in normalityTests}
count = 10
x_vals = []
while count < len(all_points):
    arr = all_points[:count]
    scores = th.plotrstatwithgaussian(arr, _bincount=50, filename='dummy', _normalitytest=normalityTests)
    x_vals.append(count)
    for k in scores:
        results[k].append(scores[k])
    count += 10
for k in normalityTests:
    plt.plot(x_vals, results[k])
    plt.title('Normality Benchmark {} test'.format(k))
    plt.xlabel('No. of samples')
    plt.ylabel('Score')
    plt.savefig('Normality-Benchmark-{}-test.png'.format(k))
    plt.clf()
