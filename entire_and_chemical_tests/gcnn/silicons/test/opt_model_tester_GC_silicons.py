#!/usr/bin/env python

import sys
sys.path.append('/srv/home/user/moleprop/util/')
import workflow as wf
import pandas as pd

print("About to simulate")
# Custom Validation (if we have the files availale)
model_args = {'nb_epoch': 150,
        'batch_size': 8,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 512,
        'dropout': 0.2,
        'number_atom_features':75,
        'learning_rate':0.0005,
        'mode': 'regression'}
metrics = ['AARD', 'RMSE', 'MAE', 'R2']
scores, predictions, test_datasets = wf.Run.custom_validation(train_dataset = '/srv/home/user/testing/full_optimization/data/optimizer_dataset.csv',
                                                              test_dataset = '/srv/home/user/testing/full_optimization/data/silicons_dataset.csv',
                                                              model = 'GC',
                                                              model_args = model_args,
                                                              metrics = metrics)

# Make plots for every fold
for key in scores:
    print(key+" = "+str(scores[key]))

# DEBUG
print("[DEBUG] len predictions: ", len(predictions))

print("About to make parity plots")
for i in range(len(predictions)):
    p_name = "parity_"+str(i)
    std = test_datasets[i]['flashpoint'].std()
    txt = {
           "RMSE":scores['RMSE_list'][i], 
           "R2":scores['R2_list'][i], 
           "MAE":scores['MAE_list'][i], 
           "AARD":scores['AARD_list'][i],
           "RMSE/std":scores['RMSE_list'][i]/std}
    wf.Plotter.parity_plot(predictions[i],test_datasets[i], plot_name = p_name, text = txt)

print("About to make residual plot")
for i in range(len(predictions)):
    r_name = "residual_"+str(i)
    std = test_datasets[i]['flashpoint'].std()
    txt = {
           "RMSE":scores['RMSE_list'][i],
           "R2":scores['R2_list'][i],
           "MAE":scores['MAE_list'][i],
           "AARD":scores['AARD_list'][i],
           "RMSE/std":scores['RMSE_list'][i]/std}
    wf.Plotter.residual_histogram(predictions[i],test_datasets[i], plot_name = r_name, text = txt)

print("About to plot full data")
P = list()                               # integration of predictions for whole dataset
for i in range(len(predictions)):
    P += predictions[i]
D = pd.concat(test_datasets)             # integration of the whole dataset
txt = {'RMSE/STD': scores['RMSE']/D['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AARD': scores['AARD']}
wf.Plotter.parity_plot(P,D,plot_name = "Full_parity", text = txt)
wf.Plotter.residual_histogram(P,D,plot_name = "Full_residual", text = txt)
