import numpy as np
import pandas as pd
from tabulate import tabulate

# Define GPR and RF errors
GPR_errors = np.random.random_sample(37, )
RF_errors = np.random.random_sample(37, )


# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data


data = importdata('_haijinlogfeatures_Pd_only.csv')

list = []
for element in data[24]:
    list.append(element)
del list[0]

list2 = []
for element in data[25]:
    list2.append(element)
del list2[0]

final_list = []
for i in range(0, 37):
    first = list[i]
    second = list2[i]
    combined = first + "-" + second
    final_list.append(combined)
print(final_list)


def predict(GPR_error, RF_error):
    if GPR_error < 0.8 and RF_error < 0.8:
        return 1
    else:
        return 0


predictions = [predict(GPR_errors[i], RF_errors[i]) for i in range(0, 37)]

results = [(final_list[i], predictions[i], GPR_errors[i], RF_errors[i]) for i in range(0, 37)]

print(tabulate(results, headers=["Material", "In domain?", "GPR predicted error", "RF predicted error"]))
