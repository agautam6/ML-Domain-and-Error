from package import io, tests

nt = tests.NormalityTests(
    normalityTests=['RMSE', 'Normalized-RMSE', 'Log-RMSE', 'Normalized-Log-RMSE'],
    trainfile='data/PVstability_Weipaper_alldata_featureselected.csv',
    rfslope=0.889069,
    rfintercept=-0.011293,
    gprsavedkernel=io.loadmodelobj(
        'models/GPR_data_PVstability_Weipaper_alldata_featureselected_csv_02-18-20_22-26-49').getGPRkernel(),
    datasetname='PV',
    data_sanitize_list=['is_testdata', 'Material Composition']
)

nt.run()
