# This script imports the Diffusion data set from its CSV file, removes unnecessary columns, and saves the x- and y-values as np arrays.
# The y-values the models will be trying to predict are from the "EnergyAboveHull (meV/atom)" column.

from package import io
import numpy as np

# set noise factor
scale = 2.0

# import data
data = io.importdata('data/Diffusion_Data_allfeatures.csv')
data = io.sanitizedata(data)

# separate x- and y-values and save as numpy arrays
X_values = data.iloc[:, 1:]
y_values = data.iloc[:, 0]
X_values = X_values.to_numpy(dtype=float)
y_values = y_values.to_numpy(dtype=float)

#generate noise
mu = 0
sigma = np.std(y_values) * scale
y_noise = np.random.normal(mu, sigma, len(y_values))

print(np.mean(y_noise))
print(np.std(y_noise))
print(np.std(y_values))

#add noise to y_values
y_values = y_values + y_noise

# save arrays for later use
np.save('Full_Method_Diffusion_noise/data/all_x_values.npy', X_values)
np.save('Full_Method_Diffusion_noise/data/all_y_values.npy', y_values)