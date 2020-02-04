from package import io
import csv


data = io.importdata('_haijinlogfeaturesnobarrier_alldata.csv')
data_features = data.iloc[:, :-4]
data_useless = data.iloc[:, -4:]
feature_fields = data_features.columns
useless_fields = data_useless.columns
data_np = data_features.to_numpy(dtype=float).std(axis=0)
filename = "NOISE_1STD_haijinlogfeaturesnobarrier_alldata.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(feature_fields.append(useless_fields))
    for j in range(0, len(data)):
        row = []
        for i in range(0, len(feature_fields)):
            row.append(float(data_features.iloc[j, i])+data_np[i])
        for i in range(len(useless_fields)):
            row.append(data_useless.iloc[j, i])
        csvwriter.writerow(row)


