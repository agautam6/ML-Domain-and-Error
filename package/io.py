import pandas as pd


# Fetch data with filename
def importdata(filename):
    data = pd.read_csv(filename, header=None, sep=',')
    data.columns = data.iloc[0]
    data = data.drop([0])
    return data


# Sanitize data. Temporarily dropping columns E_regression, Material Composition
def sanitizedata(data):
    col_list = ['Material compositions 1', 'Material compositions 2', 'E_regression']
    # Remove columns if present
    for c in col_list:
        if c in data.columns:
            data = data.drop(columns=[c])
    return data
