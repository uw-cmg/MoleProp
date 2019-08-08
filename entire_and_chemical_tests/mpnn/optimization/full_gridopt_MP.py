import sys
sys.path.append('/srv/home/user/moleprop/util/')
import os
import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
import logging
import math
import random
import pandas as pd
import integration_helpers
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from deepchem.models.tensorgraph.models.graph_models import MPNNModel

dataset_file = 'optimizer_dataset.csv'

# format optimization dataset
flashpoint_tasks = ['flashpoint']  # column name need to be exactly 'flashpoint'
loader = dc.data.CSVLoader(
    tasks=flashpoint_tasks, smiles_field="smiles", featurizer = dc.feat.ConvMolFeaturizer())
dataset = loader.featurize(dataset_file, shard_size=8192)
    
# Initialize transformers
transformers = [
    dc.trans.NormalizationTransformer(
    transform_y=True, dataset=dataset, move_mean=True)
]
for transformer in transformers:
    train_dataset = transformer.transform(dataset)

# define splitter for cross validation    
splitter = dc.splits.RandomSplitter()
train_set, valid_set = splitter.train_test_split(train_dataset, frac_train=0.8) 

# Define metric for eavluating the model by using Pearson_R2
#metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
# Define metric for eavluating the model by using RMSE
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

params_dict = {
    "nb_epoch":[70,100,150,200,400],
    "n_tasks":[1],
    "batch_size": [8,32],
    "n_atom_feat":[75],
    "n_pair_feat":[14],
    "T":[1],
    "M":[1],
    "dropout":[0.0,0.2,0.4],
    "learning_rate":[0.0005,0.001,0.005],
    "mode":["regression"]
}

def mpnn_model_builder(model_params , model_dir): 
    mp_model = MPNNModel(**model_params, model_dir = "./test_models") 
    return mp_model

print("--- starting optimization ---")
optimizer = dc.hyper.HyperparamOpt(mpnn_model_builder)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
    params_dict,
    train_set,
    valid_set,
    transformers,
    metric,
    logdir=None,
    use_max=False)  # setting use_max to be False when using RMSE as metric


print("\n===================BEST MODEL=================")
print(best_model)
print("\n===================BEST Params=================")
print(best_hyperparams)
print("\n===================ALL_RESULTS=================")

# print out all resutls 
for key in sorted(all_results.keys()):
    print(key,": ", round(all_results[key],4))
