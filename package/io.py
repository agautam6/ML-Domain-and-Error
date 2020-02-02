import pandas as pd


# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data


# Sanitize data. Temporarily dropping columns E_regression, Material Composition
def getdata(data):
    data = data.drop([24, 25, 26], axis=1)
    data = data.drop([0])
    return data


# Fetch data with filename
def importdatanames(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    return data
