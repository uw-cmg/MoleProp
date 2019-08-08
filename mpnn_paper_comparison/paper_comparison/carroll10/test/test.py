import sys
sys.path.append('../../../util')
import workflow as wf
import pandas as pd
import statistics as stat

M = 5  # number of tests we need to do

args = [
{"nb_epoch":1,
        "n_tasks":1,
        "batch_size": 8,
        "graph_conv_layers":[64,64],
        "dense_layer_size":128,
        "learning_rate":0.005,
        "mode":'regression'}
]

mpnn_args = [
	        {
	        "nb_epoch":200,
	        "n_tasks":1,
	        "batch_size":32,
            "n_atom_feat":75,
            "n_pair_feat":14,
            "T":1,
            "M":1,
            "dropout":0.2,
            "learning_rate":0.001,
            "mode": "regression"
           },
	       {
	        "nb_epoch":400,
	        "n_tasks":1,
	        "batch_size":8,
            "n_atom_feat":75,
            "n_pair_feat":14,
            "T":1,
            "M":1,
            "dropout":0.2,
            "learning_rate":0.0005,
            "mode": "regression"
           },
	       {
	        "nb_epoch":200,
	        "n_tasks":1,
	        "batch_size":8,
            "n_atom_feat":75,
            "n_pair_feat":14,
            "T":1,
            "M":1,
            "dropout":0.2,
            "learning_rate":0.0005,
            "mode": "regression"
           },
	       {
	        "nb_epoch":200,
	        "n_tasks":1,
	        "batch_size":8,
            "n_atom_feat":75,
            "n_pair_feat":14,
            "T":1,
            "M":1,
            "dropout":0.2,
            "learning_rate":0.0005,
            "mode": "regression"
           },
	       {
	        "nb_epoch":150,
	        "n_tasks":1,
	        "batch_size":32,
            "n_atom_feat":75,
            "n_pair_feat":14,
            "T":1,
            "M":1,
            "dropout":0.0,
            "learning_rate":0.001,
            "mode": "regression"
           }
     ]

scores_all = {'RMSE':[], 'MAE':[], 'R2':[], 'AAD':[]}
for i in range(M):
    print("==========Hyperparameter==========")
    print(mpnn_args[i])
    scores,pred,test_set = wf.Run.custom_validation(train_dataset = '../optimization/train_'+str(i)+'.csv',
                                                     test_dataset = '../optimization/test_'+str(i)+'.csv',
                                                     model = 'MPNN', 
                                                     model_args = mpnn_args[i],
                                                     metrics = ['MAE','RMSE','R2','AAD'])
    for key in scores_all:
        scores_all[key].append(scores[key])
    std = test_set['flashpoint'].std()
    txt = {
           "RMSE":scores['RMSE'],
           "R2":scores['R2'],
           "MAE":scores['MAE'],
           "AAD":scores['AAD'],
           "RMSE/std":scores['RMSE']/std}
    print("About to make parity plot for test ", i)
    p_name = "parity_"+str(i)
    d_name = "parity_plot_"+str(i)
    wf.Plotter.parity_plot(pred,test_set, plot_name = p_name,dir_name = d_name, text = txt)

    print("About to make residual plot for test ", i)
    r_name = "residual_"+str(i)
    d_name = "residual_plot_"+str(i)
    wf.Plotter.residual_histogram(pred,test_set, plot_name = r_name,dir_name = d_name, text = txt)
if M > 1:
    scores_mean = {'RMSE':stat.mean(scores_all['RMSE']),
                   'MAE':stat.mean(scores_all['MAE']),
                   'R2':stat.mean(scores_all['R2']),
                   'AAD':stat.mean(scores_all['AAD'])}
    scores_std = {'RMSE':stat.stdev(scores_all['RMSE']),
                   'MAE':stat.stdev(scores_all['MAE']),
                   'R2':stat.stdev(scores_all['R2']),
                   'AAD':stat.stdev(scores_all['AAD'])}
    file = open('Final_test_result.txt', 'w')
    for key in scores_mean:
        s = "mean of " + key + " = " + str(scores_mean[key]) + "\n"
        s += "std of " + key + " = " + str(scores_std[key]) + "\n\n"
        file.write(s)
    file.close()
else:
    file = open('Final_test_result.txt', 'w')
    file.write("Only one test! No mean and std!\n")
    for key in scores_all:
        s = key + " = " + str(scores_all[key][0]) + "\n"
        file.write(s)
    file.close()
