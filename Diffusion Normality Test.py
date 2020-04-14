from package import io, tests

nt = tests.NormalityTests(
    normalityTests=['RMSE', 'Normalized-RMSE', 'Log-RMSE', 'Normalized-Log-RMSE'],
    trainfile='data/Diffusion_Data_allfeatures.csv',
    rfslope=0.629705,
    rfintercept=0.037300,
    gprsavedkernel=io.loadmodelobj('models/GPR_data_Diffusion_Data_allfeatures_csv_02-24-20_18-32-12').getGPRkernel(),
    datasetname='Diffusion'
)

nt.run()
