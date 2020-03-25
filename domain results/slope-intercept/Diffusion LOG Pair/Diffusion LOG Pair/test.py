import os

from numpy import arange, meshgrid, round

gpr_thresholds_range = round(arange(0.5, 1.2, 0.1), 1)
rf_thresholds_range = round(arange(0.5, 1.2, 0.1), 1)
gpr_thresholds, rf_thresholds = meshgrid(gpr_thresholds_range, rf_thresholds_range)
accumulator = {(r, g, 1): [] for g in gpr_thresholds_range for r in rf_thresholds_range}
accumulator.update({(r, g, 0): [[], [], []] for g in gpr_thresholds_range for r in rf_thresholds_range})

# Directory setup

path = "/Users/gg/PycharmProjects/ML-Domain-and-Error/domain results/slope-intercept/Diffusion LOG Pair/Diffusion LOG Pair"

print("The current working directory is %s" % path)

for i_rf_thresholds in range(0, len(rf_thresholds_range)):
    for i_gpr_thresholds in range(0, len(gpr_thresholds_range)):
        gpr_thresh = round(gpr_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        rf_thresh = round(rf_thresholds[i_rf_thresholds, i_gpr_thresholds], 1)
        pathTemp = path + "/RF_Threshold=" + str(rf_thresh) + ",GPR_Threshold=" + str(gpr_thresh)
        print("The current working directory is %s" % pathTemp)
        try:
            os.mkdir(pathTemp)
        except OSError:
            print("Creation of the directory %s failed" % pathTemp)
        else:
            print("Successfully created the directory %s " % pathTemp)
