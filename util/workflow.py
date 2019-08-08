import os
import sys
import numpy as np
import pandas as pd
import random                                 # for randomly choosing indices from left-out group as test indices
import statistics as stat
import integration_helpers                               # for removing duplicates
import itertools
import tempfile
import shutil
import collections
import tensorflow as tf
import deepchem as dc
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import integration_helpers                               # for removing duplicates
plt.switch_backend('agg')
#plt.rc('font', size = 8)                                # change plot font size
#matplotlib.use('agg')
from functools import reduce
from operator import mul
from sklearn.model_selection import KFold
from deepchem.trans.transformers import undo_transforms  # for getting real predictions
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.save import log


class Loader:
    def load(file_name, data_dir = './', getinfo = True):
        """
        load data from .csv file
        """
        # TODO: need to clean data before loading?
        data_file = os.path.join(data_dir, file_name)
        if not os.path.exists(data_file):
            if data_dir == './':
                data_dir = "current"
            error_msg = file_name + " was not found in " + data_dir + " directory"
            sys.exit(error_msg)
        print("|||||||||||||||||||||Loading " + file_name+ "|||||||||||||||||||||||")
        data = pd.read_csv(data_file) # encoding='latin-1' might be needed
        if getinfo:
            Loader.getinfo(data, 'Original_dataset')
        return data

    def getinfo(data, name = 'getinfo',dir_name = 'dataset_info'):
        """
            name: .txt file name
            dir_name: dir name
        get information of the dataset
        """
        # Create target Directory if don't exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("||||||||||||||||Directory dataset_info Created||||||||||||||||")
        output = ("=====================================================\n" +
                  name + " info:\n"+
                  "-----------------------------------------------------\n" +
                  "Dataset length is: " + str(len(data.index)) + "\n")
        if 'source' in data.columns:
            sources = data.source.unique()
            source_info = dict()
            for s in sources:
                temp = data.loc[data.source == s]
                source_info[s] = [len(temp)]
            output += ("-----------------------------------------------------\n" +
                     "Dataset sources info: \n")
            for s in sources:
                output += (str("  Source name:" + str(s) + ", Number of data: " + str(source_info[s]) + "\n"))

        output += ("-----------------------------------------------------\n" +
                   "Mean: " + str(data['flashpoint'].mean()) + "\n"
                   "-----------------------------------------------------\n" +
                   "Std: " + str(data['flashpoint'].std()) + "\n" +
                   "=====================================================\n")
        print(output)
        file = open('./'+dir_name+'/'+name+'.txt', 'w')
        file.write(output)
        file.close()
        
    def get_single_dataset(integrated_dataset, source_name):
        """
        this method is used to extract dataset from intgrated dataset

        integrated_dataset: DataFrame
        source_name: name of the source you want to extract from the integrated dataset
        return: DataFrame
        """
        return integrated_dataset.loc[integrated_dataset['source']==source_name]

class Splitter:
    def k_fold_source(dataset, source, n_splits = 3, shuffle = True, random_state = None):
        dataset_source = dataset[dataset['source'] == source]
        indices, data = Splitter.k_fold(dataset_source, n_splits, shuffle, random_state)
        return indices, data

    def k_fold(dataset, n_splits = 3, shuffle = True, random_state = None, getinfo = True):
        """
        split data into k-fold
        Return:
        indices of k-fold training and test sets
        new dataset after removing duplicates
        """
        dataset = integration_helpers.remove_duplicates(dataset) # remove duplicates
        if getinfo:
            Loader.getinfo(dataset, 'Full_dataset')
        if shuffle == True:
            random_state = 4396
        kf = KFold(n_splits, shuffle, random_state)
        indices = kf.split(dataset)
        return indices, dataset

    def LOG(dataset,
            test_group,
            use_metallics = None,
            use_silicons = None,
            use_tin = None,
            use_acids = None,
            n_splits = None,
            transfer_learning = None,
            tl_n_splits = None,
            frac = 1,
           getinfo = True):  # leave out group
        """
        split dataset by leaving out a specific source as test set
    
        params:
        dataset: data frame
        test_group: string
        frac: fraction of the left-out group that will be used as test set
        n_splits: n-fold CV: leave 1/n of the test_group as test set, 
                             the remaining 80% will be added to the training set. 
                  If None, do single validation
        transfer_learning: flag for transfer learning
        tl_n_splits: n-fold CV for transfer learning: leave 1/n of the test_group as test set, 
                                                      the remaining 80% will be used as the second training set
                     If none, based on frac to do one transfer leaning validaiton
        """
        # remove duplicates in train group.
        if not use_metallics and not use_silicons and not use_tin and not use_acids:
          print("using group to split datasets")
          test_df = dataset[dataset['source'] == test_group]
          train_df = dataset[dataset['source'] != test_group]
          print("||||||||||||||||||| "+test_group+ " will be used as test set|||||||||||||||||||")
        elif use_metallics and not use_silicons and not use_tin and not use_acids:
          print("using metallics as transfer target")
          test_df = dataset[dataset['is_metallic'] == 1]
          train_df = dataset[dataset['is_metallic'] != 1]
          print("||||||||||||||||||| metallics will be used as test set|||||||||||||||||||")
        elif not use_metallics and use_silicons and not use_tin and not use_acids:
          print("using silicons as transfer target")
          test_df = dataset[dataset['is_silicon'] == 1]
          train_df = dataset[dataset['is_silicon'] != 1]
          print("||||||||||||||||||| silicons will be used as test set|||||||||||||||||||")
        elif not use_metallics and not use_silicons and use_tin and not use_acids:
          print("using tins as transfer target")
          test_df = dataset[dataset['is_tin'] == 1]
          train_df = dataset[dataset['is_tin'] != 1]
          print("||||||||||||||||||| tin will be used as test set|||||||||||||||||||")
        elif not use_metallics and not use_silicons and not use_tin and use_acids:
          print("using acids as transfer target")
          test_df = dataset[dataset['is_acid'] == 1]
          train_df = dataset[dataset['is_acid'] != 1]
          print("||||||||||||||||||| acids will be used as test set|||||||||||||||||||")
        else: # always default to silicons
          print("using silicons as transfer target")
          test_df = dataset[dataset['is_silicon'] == 1]
          train_df = dataset[dataset['is_silicon'] != 1]
          print("||||||||||||||||||| silicons will be used as test set|||||||||||||||||||")
    
        #remove dups in training set
        train_df = integration_helpers.remove_duplicates(train_df)
        test_df = integration_helpers.remove_duplicates(test_df)
    
        # remove data points in  train dataframe that match smiles strings in
        # test dataframe
        for index, row in test_df.iterrows():
            smi = row['smiles']
            train_df = train_df[train_df['smiles'] != smi]
        frames = [train_df, test_df]
        dataset = pd.concat(frames)
        dataset.reset_index(drop=True, inplace=True)
        if getinfo:
            Loader.getinfo(dataset, 'Full_dataset')
        raw_test_indices = []
        raw_train_indices = list(range(len(dataset.index)))
        if not use_metallics and not use_silicons and not use_tin and not use_acids:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['source'] == test_group:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        elif use_metallics and not use_silicons and not use_tin and not use_acids:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['is_metallic'] == 1:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        elif not use_metallics and use_silicons and not use_tin and not use_acids:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['is_silicon'] == 1:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        elif not use_metallics and not use_silicons and use_tin and not use_acids:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['is_tin'] == 1:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        elif not use_metallics and not use_silicons and not use_tin and use_acids:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['is_acid'] == 1:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        else:
            for i in range(len(dataset.index)):
                if dataset.iloc[i]['is_silicon'] == 1:
                    raw_test_indices.append(i)
                    raw_train_indices.remove(i)
        indices = []
        if transfer_learning == None:
            if n_splits == None:
                test_indices = random.sample(raw_test_indices, int(frac*len(raw_test_indices)))
                raw_test_indices = [x for x in raw_test_indices if x not in test_indices]
                train_indices = raw_train_indices + raw_test_indices
                indices.append((train_indices, test_indices))
            else:
                test_chunk = []
                n = int(len(raw_test_indices)/n_splits)
                split_num = 0
                for i in range(0,len(raw_test_indices),n):
                    split_num +=1
                    if split_num == n_splits: 
                        # to make sure the additional several data at the end will not be generated as a new fold
                        test_chunk = raw_test_indices[i:]
                        i = len(raw_test_indices)+1
                    else:
                        test_chunk = raw_test_indices[i:i+n]
                    train = raw_train_indices+ [ind for ind in raw_test_indices if ind not in test_chunk]
                    test = test_chunk
                    indices.append((train,test))
                    if split_num == n_splits:
                        break
        else: # for transfer learning
            if tl_n_splits == None:
                test_indices = random.sample(raw_test_indices, int(frac*len(raw_test_indices)))
                second_train_indices = [x for x in raw_test_indices if x not in test_indices]
                train_indices = raw_train_indices
                test_indices = test_indices
                indices.append((train_indices, test_indices, second_train_indices))
            else:
                n = int(len(raw_test_indices)/tl_n_splits)
                split_num = 0
                for i in range(0,len(raw_test_indices),n):
                    split_num +=1
                    if split_num == tl_n_splits: 
                        # to make sure the additional several data at the end will not be generated as a new fold
                        test_indices = raw_test_indices[i:]
                    else:
                        test_indices = raw_test_indices[i:i+n]
                    second_train_indices = [ind for ind in raw_test_indices if ind not in test_indices]
                    train_indices = raw_train_indices
                    test_indices = test_indices
                    indices.append((train_indices,test_indices, second_train_indices))
                    if split_num == tl_n_splits:
                        break                
        return indices, dataset

    def subset_sample(dataset, size, n_splits=3):
      perm = np.random.RandomState(seed=size).permutation(dataset.index)
      random_data = dataset.iloc[perm]
      print('random sample size')
      print(random_data.shape)
      return Splitter.k_fold(random_data, n_splits)
      #perm = np.random.RandomState(seed=size).permutation(dataset.index)
      #m = len(dataset.index)
      #test_end = m
      #train_end = m - size
      #train_df = dataset.ix[perm[:train_end]]
      #test_df = dataset.ix[perm[train_end:test_end]]

      #for index, row in test_df.iterrows():
      #    test_df.loc[index,'source'] = "random"

      #frames = [train_df, test_df]
      #dataset = pd.concat(frames)
      #dataset.reset_index(drop=True, inplace=True)

      return dataset

    def get_organosilicons(dataset):
        train_df = dataset.copy()
        for idx, r in dataset.iterrows():
          smi = r['smiles']
          if "Si" not in smi:
            train_df.drop(idx, inplace=True)
        return train_df

    def get_organometallics(dataset):
        metallic_strs = [
            "bery",
            "Bery",
            "BERY",
            "Na", # but not "Nap"
            "Mg",
            "alu",
            "Alu",
            "ALU",
            "K+",
            "calc",
            "Calc",
            "CALC",
            "tit",
            "Tit",
            "TIT",
            "chromi",
            "Chromi",
            "CHROMI",
            "Mn",
            "Fe",
            "nick",
            "Nick",
            "NICK",
            "copp",
            "Copp",
            "COPP",
            "Zn",
            "gall",
            "Gall",
            "GALL",
            "Zr",
            "Nb",
            "Ag",
            "Cd",
            "Sn",
            "Hf",
            "tant",
            "Tant",
            "TANT",
            "Hg",
            "Tl",
            "Pb",
            "bor", # but not "born"
            "Bor",
            "BOR",
            "Si",
            "Ge",
            "ars",
            "Ars",
            "ARS"
            ]
        train_df = dataset.copy()
        found = False
        for idx, r in dataset.iterrows():
          found = False
          smi = r['compound'] + ',' + r['smiles']
          for m in metallic_strs:
            #print("smi: ", smi)
            #print("m: ", m)
            if m in smi:
              if "Na" == m and "Nap" in smi:
                break
              if ("bor" == m or "BOR" == m or "Bor" == m) and ("born" in smi or "BORN" in smi or "Born" in smi):
                break
              found = True
              break
          if not found:
            train_df.drop(idx, inplace=True)
        return train_df

    def leave_out_moleClass(dataset, mole_class_to_leave_out):
        """
        TODO:
        leave out a specific mole class
        """
        return 0

class Run:
    def cv(data,
           indices, # k-fold indices of training and test sets
           model,  # need to be either MPNN or GraphConv
           model_args = None,
           metrics = None,
           n_splits = 3,
           model_obj = None):
        """
        pass data into models (MPNN or graphconv) and conduct cross validation

        Return:
        avg_cv_rms_score
        avg_cv_mae_score
        cv_rms_scores: a list of RMSE scores from cross validation
        cv_mae_scores: a list of MAE scores from cross validation
        """
        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv' or model == 'weave'):
            sys.exit("Only support MPNN model and GraphConv and weave model")
        cv_rms_scores = []
        cv_mae_scores = []
        cv_r2_scores = []
        cv_aad_scores = []
        cv_predictions = []
        cv_test_datasets = []
        cv_train_scores = []
        outliers = list()
        i = 1       # track the number of iteration
        for train_indices, test_indices in indices:
            train_set = data.iloc[train_indices]
            test_set = data.iloc[test_indices]
            cv_test_datasets.append(test_set)
            train_set.to_csv('train_set.csv',index = False)
            test_set.to_csv('test_set.csv',index = False)
            if model == 'MPNN':
                rms_score,mae_score,r2_score,train_scores,pred = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
            elif model == 'graphconv' or model == 'GC' or model == 'GraphConv':
                rms_score,mae_score,r2_score,train_scores,pred = Model.graphconv(model_args,"train_set.csv", "test_set.csv")
            elif model == 'weave':
                rms_score,mae_score,r2_score,train_scores,pred = Model.weave(model_args,"train_set.csv", "test_set.csv") 
            Loader.getinfo(train_set, 'CV_'+str(i)+"_Train")
            Loader.getinfo(test_set, 'CV_'+str(i)+"_Test")
            i += 1
            cv_train_scores.append(train_scores)
            cv_rms_scores.append(rms_score)
            cv_mae_scores.append(mae_score)
            cv_r2_scores.append(r2_score)
            cv_aad_scores.append((Run.getAAPD(test_set,pred)))
            cv_predictions.append(pred)
            outliers.append(Run.get_outliers(test_set, pred))
            os.remove("train_set.csv")
            os.remove("test_set.csv")
        avg_rms_score = sum(cv_rms_scores)/n_splits
        avg_mae_score = sum(cv_mae_scores)/n_splits
        avg_r2_score = sum(cv_r2_scores)/n_splits
        avg_aad_score = sum(cv_aad_scores)/n_splits
        # calculate avg train scores
        avg_train_scores = (sum(cv_train_scores[i][0] for i in range(n_splits))/n_splits,
                            sum(cv_train_scores[i][1] for i in range(n_splits))/n_splits)
        scores_all = {'RMSE':avg_rms_score,'RMSE_list':cv_rms_scores,
                      'MAE': avg_mae_score,'MAE_list':cv_mae_scores,
                      'R2': avg_r2_score, 'R2_list': cv_r2_scores,
                      'AAD':avg_aad_score, 'AAD_list': cv_aad_scores,
                     'train':avg_train_scores}
        scores = dict()
        if metrics == None:  # return default scores (RMSE and R2)
            scores = {'RMSE':scores_all['RMSE'],
                      'R2':scores_all['R2'],
                      'RMSE_list':scores_all['RMSE_list'],
                      'R2_list':scores_all['R2_list'],
                     'train': scores_all['train']}
        else:
            for m in metrics:
                if not ( m == 'RMSE' or m == 'MAE' or m == 'AAD' or m == 'R2' or m == 'train'):
                    sys.exit('only supports RMSE, MAE, AAD, AAE, R2, and train')
                scores[m] = scores_all[m]
                if m != 'train':
                    list_name = str(m + '_list')
                    scores[list_name] = scores_all[list_name]
        outliers = pd.concat(outliers)
        outliers.to_csv('outliers.csv')
        file = open('FINAL_RESULT.txt', 'w')
        for key in scores:
            s = key + " = " + str(scores[key])+"\n"
            file.write(s)
        #file.write(str(scores))
        file.close()
        return scores, cv_predictions, cv_test_datasets
    
    def LOG_validation(data,
                       indices,
                       model,
                       model_args = None,
                       metrics = None):
        """
        Conduct leave-out-group validation
        """

        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv' or model == 'weave'):
            sys.exit("Only supports MPNN model and graphconv and weave model")
        train_indices, test_indices = indices[0]
        train_set = data.iloc[train_indices]
        test_set = data.iloc[test_indices]
        train_set.to_csv('train_set.csv',index = False)
        test_set.to_csv('test_set.csv',index = False)
        Loader.getinfo(train_set, "LOG_Train")
        Loader.getinfo(test_set, "LOG_Test")
        if model == 'MPNN':
            rms_score,mae_score,r2_score,train_scores,pred = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
        elif model == 'GraphConv' or model == 'graphconv' or model == 'GC':
            rms_score,mae_score,r2_score,train_scores,pred = Model.graphconv(model_args,"train_set.csv", "test_set.csv")
        elif model == 'weave':
            rms_score,mae_score,r2_score,train_scores,pred = Model.weave(model_args,"train_set.csv", "test_set.csv")
        os.remove("train_set.csv")
        os.remove("test_set.csv")
        scores_all = {'RMSE':rms_score,
                      'MAE': mae_score,
                      'R2': r2_score,
                      'AAD':(Run.getAAPD(test_set,pred))}
        scores = dict()
        if metrics == None:  # return default scores (RMSE and R2)
            scores = {'RMSE':scores_all['RMSE'],
                      'R2':scores_all['R2']}
        else:
            for m in metrics:
                if not ( m == 'RMSE' or m == 'MAE' or m == 'AAD' or m == 'R2' or m == 'train'):
                    sys.exit('only supports RMSE, MAE, AAD, AAE, and R2 and train')
                scores[m] = scores_all[m]
        outliers = Run.get_outliers(test_set, pred)
        outliers.to_csv('outliers.csv')
        file = open('FINAL_RESULT.txt', 'w')
        for key in scores:
            s = key + " = " + str(scores[key]) + "\n"
            file.write(s)
        file.close()
        return scores, pred, test_set

    def custom_validation(train_dataset,
                          test_dataset,
                          model, 
                          model_args = None,
                          metrics = None):
        """
        Do custom validation:
        use customized training and test sets for our models now
        train_dataset & test_dataset: path + name of csv files, string
        """
        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv'):
            sys.exit("Only supports MPNN model and graphconv model")
        train_set = Loader.load(train_dataset)
        test_set = Loader.load(test_dataset)
        if model == 'MPNN':
            rms_score,mae_score,r2_score,train_scores,pred = Model.MPNN(model_args, train_dataset, test_dataset)
        elif model == 'GraphConv' or model == 'graphconv' or model == 'GC':
            rms_score,mae_score,r2_score,train_scores,pred = Model.graphconv(model_args,train_dataset, test_dataset)
        elif model == 'weave':
            rms_score,mae_score,r2_score,train_scores,pred = Model.weave(model_args,train_dataset, test_dataset)
        scores_all = {'RMSE':rms_score,
                      'MAE': mae_score,
                      'R2': r2_score,
                      'AAD':(Run.getAAPD(test_set,pred)),
                     'train':train_scores}
        scores = dict()
        if metrics == None:  # return default scores (RMSE and R2)
            scores = {'RMSE':scores_all['RMSE'], 
                      'R2':scores_all['R2'],
                      'MAE':scores_all['MAE'],
                     'train':scores_all['train']}
        else:
            for m in metrics:
                if not ( m == 'RMSE' or m == 'MAE' or m == 'AAD' or m == 'R2' or m =='train'):
                    sys.exit('only supports RMSE, MAE, AAD, AAE, R2, and train')
                scores[m] = scores_all[m]
        outliers = Run.get_outliers(test_set, pred)
        outliers.to_csv('outliers.csv')
        return scores, pred, test_set

    def basic_transfer(data,
                       indices,
                       model,
                       model_args = None,
                       metrics = None,
                       do_cv = False,
                       n_splits = 5):
        """
        Conduct basic transfer learning
          - fit on non-target dataset
          - save model
          - continue training on target dataset
          - run validation tests
        """

        # DEBUG
        print("[DEBUG] in basic_transfer")

        # check inputs
        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv'):
            sys.exit("Only supports MPNN model and graphconv model")

        train_indices = indices[0][0]
        train_set = data.iloc[train_indices]
        train_set.to_csv('train_set.csv',index = False)
        Loader.getinfo(train_set, "LOG_Train")

        # get model
        model_obj = None
        if model == 'GraphConv' or model == 'graphconv' or model == 'GC':
            model_obj = Model.graphconv(model_args,"train_set.csv", "train_set.csv", True)
        elif model == 'MPNN':
            model_obj = Model.MPNN(model_args,"train_set.csv", "train_set.csv", True)
        else:
          print("you messed up boi")
          return

        # DEBUG
        print("[DEBUG] model object built, trained, and saved")

        # continue training, then test
        if not do_cv:
          # split data
          test_indices = indices[0][1]
          second_train_indices = indices[0][2]
          test_set = data.iloc[test_indices]
          second_train_set = data.iloc[second_train_indices]
          test_set.to_csv('test_set.csv',index = False)
          second_train_set.to_csv('second_train_set.csv',index = False)
          Loader.getinfo(test_set, "LOG_Test")
          Loader.getinfo(second_train_set, "LOG_SecondTrain")

          # DEBUG
          print("[DEBUG] in basic_transfer")
          print("[DEBUG] size of train_set: ", len(train_set.index))
          print("[DEBUG] size of test_set: ", len(test_set.index))
          print("[DEBUG] size of second_train_set: ", len(second_train_set.index))

          if model == 'GraphConv' or model == 'graphconv' or model == 'GC':
              rms_score,mae_score,r2_score,train_scores,pred = Model.update_model_gc(model_obj,"second_train_set.csv", "test_set.csv")
          elif model == 'MPNN':
              rms_score,mae_score,r2_score,train_scores,pred = Model.update_model_mp(model_obj,"second_train_set.csv", "test_set.csv")

          # DEBUG
          print("[DEBUG] model updated and tested")

          # collect results
          os.remove("train_set.csv")
          os.remove("test_set.csv")
          os.remove("second_train_set.csv")
          rms_scores = []
          rms_scores.append(rms_score)
          mae_scores = []
          mae_scores.append(mae_score)
          r2_scores = []
          r2_scores.append(r2_score)
          aad_scores = []
          aad_score = (Run.getAAPD(test_set,pred))
          aad_scores.append(aad_score)
          scores_all = {'RMSE':rms_score,'RMSE_list':rms_scores,
                        'MAE': mae_score,'MAE_list':mae_scores,
                        'R2': r2_score,'R2_list': r2_scores,
                        'AAD':aad_score,'AAD_list':aad_scores}
          scores = dict()
          if metrics == None:  # return default scores (RMSE and R2)
              scores = {'RMSE':scores_all['RMSE'],
                        'R2':scores_all['R2'],
                        'RMSE_list':scores_all['RMSE_list'],
                        'R2_list':scores_all['R2_list']}
          else:
              for m in metrics:
                  if not ( m == 'RMSE' or m == 'MAE' or m == 'AAD' or m == 'R2'):
                      sys.exit('only supports RMSE, MAE, AAD, AAE, and R2')
                  scores[m] = scores_all[m]
                  list_name = str(m + '_list')
                  scores[list_name] = scores_all[list_name]
          outliers = Run.get_outliers(test_set, pred)
          outliers.to_csv('outliers.csv')
          file = open('FINAL_RESULT.txt', 'w')
          for key in scores:
              s = key + " = " + str(scores[key]) + "\n"
              file.write(s)
          file.close()
          predictions = []
          predictions.append(pred)
          test_datasets = []
          test_datasets.append(test_set)
        else:
          latest_checkpt_file = tf.train.latest_checkpoint("./models")
          cv_rms_scores = []
          cv_mae_scores = []
          cv_r2_scores = []
          cv_aad_scores = []
          cv_predictions = []
          cv_test_datasets = []
          cv_train_scores = []
          outliers = list()
          i = 1       # track the number of iteration
          for train_indices, test_indices, second_train_indices in indices:
              train_set = data.iloc[second_train_indices]
              test_set = data.iloc[test_indices]
              cv_test_datasets.append(test_set)
              train_set.to_csv('train_set.csv',index = False)
              test_set.to_csv('test_set.csv',index = False)
              model_obj.restore(latest_checkpt_file)
              if model == 'MPNN':
                  rms_score,mae_score,r2_score,train_scores,pred = Model.update_model_mp(model_obj, "train_set.csv", "test_set.csv")
              elif model == 'graphconv' or model == 'GC' or model == 'GraphConv':
                  rms_score,mae_score,r2_score,train_scores,pred = Model.update_model_gc(model_obj,"train_set.csv", "test_set.csv")
              Loader.getinfo(train_set, 'CV_'+str(i)+"_Train")
              Loader.getinfo(test_set, 'CV_'+str(i)+"_Test")
              i += 1
              cv_train_scores.append(train_scores)
              cv_rms_scores.append(rms_score)
              cv_mae_scores.append(mae_score)
              cv_r2_scores.append(r2_score)
              cv_aad_scores.append((Run.getAAPD(test_set,pred)))
              cv_predictions.append(pred)
              outliers.append(Run.get_outliers(test_set, pred))
              os.remove("train_set.csv")
              os.remove("test_set.csv")
          avg_rms_score = sum(cv_rms_scores)/n_splits
          avg_mae_score = sum(cv_mae_scores)/n_splits
          avg_r2_score = sum(cv_r2_scores)/n_splits
          avg_aad_score = sum(cv_aad_scores)/n_splits
          # calculate avg train scores
          avg_train_scores = (sum(cv_train_scores[i][0] for i in range(n_splits))/n_splits,
                              sum(cv_train_scores[i][1] for i in range(n_splits))/n_splits)
          scores_all = {'RMSE':avg_rms_score,'RMSE_list':cv_rms_scores,
                        'MAE': avg_mae_score,'MAE_list':cv_mae_scores,
                        'R2': avg_r2_score, 'R2_list': cv_r2_scores,
                        'AAD':avg_aad_score, 'AAD_list': cv_aad_scores,
                       'train':avg_train_scores}
          scores = dict()
          if metrics == None:  # return default scores (RMSE and R2)
              scores = {'RMSE':scores_all['RMSE'],
                        'R2':scores_all['R2'],
                        'RMSE_list':scores_all['RMSE_list'],
                        'R2_list':scores_all['R2_list'],
                       'train': scores_all['train']}
          else:
              for m in metrics:
                  if not ( m == 'RMSE' or m == 'MAE' or m == 'AAD' or m == 'R2' or m == 'train'):
                      sys.exit('only supports RMSE, MAE, AAD, AAE, R2, and train')
                  scores[m] = scores_all[m]
                  if m != 'train':
                      list_name = str(m + '_list')
                      scores[list_name] = scores_all[list_name]
          outliers = pd.concat(outliers)
          outliers.to_csv('outliers.csv')
          file = open('FINAL_RESULT.txt', 'w')
          for key in scores:
              s = key + " = " + str(scores[key])+"\n"
              file.write(s)
          #file.write(str(scores))
          file.close()
          predictions = cv_predictions
          test_datasets = cv_test_datasets

        # DEBUG
        print("[DEBUG] model updated and tested")

        return scores, predictions, test_datasets
          
    def getAAPD(dataset, pred):  # Average absolute percent deviation
        expt = dataset['flashpoint'].tolist()
        sum = 0
        for i in range(len(dataset)):
            sum += (abs(expt[i] - pred[i])/expt[i])
        sum = sum*100/len(dataset)
        return sum

    def get_outliers(dataset,pred):
        expt = dataset['flashpoint'].tolist()
        residual = list()
        for i in range(len(dataset)):
            residual.append((expt[i]-pred[i]))
        dataset['residual'] = residual
        outliers = dataset[(dataset.residual > 100) | (dataset.residual < -100)]
        return outliers


class Model:
    """
    return results for all metrics
    """
    default_args = {
        'graphconv': {
            'nb_epoch': 100, 
            'batch_size': 128, 
            'n_tasks': 1,
            'graph_conv_layers':[64,64],
            'dense_layer_size': 128,
            'dropout': 0,
            'mode': 'regression',
            'learning_rate': 0.0005},
        'MPNN':{
            'n_tasks':1,
            'n_atom_feat':75,
            'n_pair_feat':14,      # NEED to be 14 for WaveFeaturizer
            'T':1,
            'M':1,
            'batch_size':16,
            'nb_epoch': 50,
            'learning_rate':0.0001,
            'use_queue':False,
            'mode':"regression"},
        'weave':{
            'learning_rate':0.0005,
            'n_tasks':1,
            'n_atom_feat':75,
            'n_pair_feat':14,
            'n_hidden':50,
            'n_graph_feat':128,
            'mode':"regression",
            'batch_size':64,
            'nb_epoch':40}
    }

    def graphconv(args, train_set, test_set, only_model = False):
        # parse arguments
        model_args = Model.default_args['graphconv']
        if args != None:
            for key in args:
                model_args[key] = args[key]
        flashpoint_tasks = ['flashpoint']  # Need to set the column name to be excatly "flashpoint"
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = dc.feat.ConvMolFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
            test_dataset = transformer.transform(test_dataset)
        if only_model:
          # DEBUG
          print("[DEBUG] in only_model mode")

          model = dc.models.GraphConvModel(n_tasks = model_args['n_tasks'],
                                           mode = model_args['mode'],
                                           dropout = model_args['dropout'],
                                           model_dir='models',
                                           learning_rate = model_args['learning_rate'])
          model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
          model.save()
          return model
        else:
          model = dc.models.GraphConvModel(n_tasks = model_args['n_tasks'],
                                           mode = model_args['mode'],
                                           dropout = model_args['dropout'],
                                           learning_rate = model_args['learning_rate'])
          metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
          metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
          metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean) # R2 score
          model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
          pred = model.predict(test_dataset)
          pred = undo_transforms(pred, transformers)

          output_stuff = []
          untransformed_test_labels = undo_transforms(test_dataset.y, transformers)
          for idx,y in enumerate(untransformed_test_labels):
            output_stuff.append((y[0],pred[idx][0]))
          np.savetxt("predictions.csv", output_stuff, delimiter=",")

          flattened_pred = [y for x in pred for y in x]    # convert list of lists to faltten list
          rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
          mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
          r2_score =  list( model.evaluate(test_dataset, [metric_r2], transformers).values()).pop()
          train_rms_score = list( model.evaluate(train_dataset, [metric_rms],transformers).values()).pop()
          train_r2_score =  list( model.evaluate(train_dataset, [metric_r2], transformers).values()).pop()
          return rms_score, mae_score, r2_score, (train_rms_score,train_r2_score),flattened_pred

    def MPNN(args, train_set, test_set, only_model = False):
        # parse arguments
        model_args = Model.default_args['MPNN']
        if args != None:
            for key in args:
                model_args[key] = args[key]
        flashpoint_tasks = ['flashpoint']
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = dc.feat.WeaveFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
            test_dataset = transformer.transform(test_dataset)
        if only_model:
          # DEBUG
          print("[DEBUG] in only_model mode")

          model = dc.models.MPNNModel(n_tasks = model_args['n_tasks'],
                                      n_atom_feat = model_args['n_atom_feat'],
                                      n_pair_feat = model_args['n_pair_feat'],
                                      T = model_args['T'],
                                      M = model_args['M'],
                                      batch_size = model_args['batch_size'],
                                      learning_rate = model_args['learning_rate'],
                                      use_queue = model_args['use_queue'],
                                      mode = model_args['mode'])
          model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
          model.save()
          return model
        else:
          model = dc.models.MPNNModel(n_tasks = model_args['n_tasks'],
                                      n_atom_feat = model_args['n_atom_feat'],
                                      n_pair_feat = model_args['n_pair_feat'],
                                      T = model_args['T'],
                                      M = model_args['M'],
                                      batch_size = model_args['batch_size'],
                                      learning_rate = model_args['learning_rate'],
                                      use_queue = model_args['use_queue'],
                                      mode = model_args['mode'])
          metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
          metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
          metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean) # R2 score
          model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
          pred = model.predict(test_dataset)
          pred = undo_transforms(pred, transformers)
          flattened_pred = [y for x in pred for y in x]    # convert list of lists to faltten list
          rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
          mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
          r2_score = list( model.evaluate(test_dataset, [metric_r2],transformers).values()).pop()
          train_rms_score = list( model.evaluate(train_dataset, [metric_rms],transformers).values()).pop()
          train_r2_score =  list( model.evaluate(train_dataset, [metric_r2], transformers).values()).pop()
          return rms_score, mae_score, r2_score, (train_rms_score,train_r2_score),flattened_pred
    
    def weave(args, train_set, test_set):
        # parse arguments
        model_args = Model.default_args['weave']
        if args != None:
            for key in args:
                model_args[key] = args[key]
        flashpoint_tasks = ['flashpoint']  # Need to set the column name to be excatly "flashPoint"
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = dc.feat.WeaveFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
            test_dataset = transformer.transform(test_dataset)
        model = dc.models.WeaveModel(n_tasks = model_args['n_tasks'],
                                    n_atom_feat = model_args['n_atom_feat'],
                                    n_pair_feat = model_args['n_pair_feat'],
                                    n_graph_feat = model_args['n_atom_feat'],
                                    n_hidden = model_args['n_hidden'],
                                    batch_size = model_args['batch_size'],
                                    learning_rate = model_args['learning_rate'],
                                    mode = model_args['mode'])
        metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
        metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
        metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean) # R2 score
        model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
        pred = model.predict(test_dataset)
        pred = undo_transforms(pred, transformers)
        flattened_pred = [y for x in pred for y in x]    # convert list of lists to faltten list
        rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
        mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
        r2_score =  list( model.evaluate(test_dataset, [metric_r2], transformers).values()).pop()
        train_rms_score = list( model.evaluate(train_dataset, [metric_rms],transformers).values()).pop()
        train_r2_score =  list( model.evaluate(train_dataset, [metric_r2], transformers).values()).pop()
        return rms_score, mae_score, r2_score, (train_rms_score,train_r2_score),flattened_pred

    def update_model_gc(model, train_set, test_set):
        # DEBUG
        print("[DEBUG] in update_model_gc")

        # data processing
        model_args = Model.default_args['graphconv']
        flashpoint_tasks = ['flashpoint']  # Need to set the column name to be excatly "flashpoint"
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = dc.feat.ConvMolFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
            test_dataset = transformer.transform(test_dataset)     # Modified by Sean 5/6/2019

        # setup metrics
        metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
        metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
        metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean) # R2 score

        # continue fitting
        model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])

        # predict
        pred = model.predict(test_dataset)
        pred = undo_transforms(pred, transformers)

        output_stuff = []
        untransformed_test_labels = undo_transforms(test_dataset.y, transformers)
        for idx,y in enumerate(untransformed_test_labels):
          output_stuff.append((y[0],pred[idx][0]))
        np.savetxt("predictions.csv", output_stuff, delimiter=",")

        flattened_pred = [y for x in pred for y in x]    # convert list of lists to faltten list
        rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
        mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
        r2_score =  list( model.evaluate(test_dataset, [metric_r2], transformers).values()).pop()
        train_rms_score = list( model.evaluate(train_dataset, [metric_rms],transformers).values()).pop()
        train_r2_score =  list( model.evaluate(train_dataset, [metric_r2], transformers).values()).pop()
        return rms_score, mae_score, r2_score, (train_rms_score,train_r2_score),flattened_pred

    def update_model_mp(model, train_set, test_set):
        # DEBUG
        print("[DEBUG] in update_model_mp")

        # parse arguments
        model_args = Model.default_args['MPNN']
        flashpoint_tasks = ['flashpoint']
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = dc.feat.WeaveFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
            test_dataset = transformer.transform(test_dataset)

        # setup metrics
        metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
        metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
        metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean) # R2 score

        # continue fitting
        model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])

        # predict
        pred = model.predict(test_dataset)
        pred = undo_transforms(pred, transformers)

        output_stuff = []
        untransformed_test_labels = undo_transforms(test_dataset.y, transformers)
        for idx,y in enumerate(untransformed_test_labels):
          output_stuff.append((y[0],pred[idx][0]))
        np.savetxt("predictions.csv", output_stuff, delimiter=",")

        flattened_pred = [y for x in pred for y in x]    # convert list of lists to faltten list
        rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
        mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
        r2_score = list( model.evaluate(test_dataset, [metric_r2],transformers).values()).pop()
        train_rms_score = list( model.evaluate(train_dataset, [metric_rms],transformers).values()).pop()
        train_r2_score =  list( model.evaluate(train_dataset, [metric_r2], transformers).values()).pop()
        return rms_score, mae_score, r2_score, (train_rms_score,train_r2_score),flattened_pred

class Plotter:
    def parity_plot(pred, test_dataset, errorbar = False, plot_name = 'parity_plot',dir_name = 'parity_plot',text = None):
        """
        text: dict of text that you want to add to the plot
        pred: List - predicted results
        test_dataset: DataFrame - original test dataset containing index, true flashpoints, source
        errorbar: if true, plot scatter plot for error bars
        """
        # Create target Directory if don't exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("||||||||||||||||Directory parity_plot Created||||||||||||||||")
        # add pred_result to the test_dataset DataFrame
        test_dataset = pd.DataFrame(test_dataset)
        seaborn.set(style = 'white',font_scale = 2)
        yeer = []
        avg_pred = []
        if errorbar == True:
            for i in range(len(test_dataset)):
                yeer.append(max(pred[i]) - min(pred[i]))
                avg_pred.append(stat.mean(pred[i]))
        else:
            avg_pred = pred
            yeer = [0]*len(test_dataset)
        test_dataset['pred'] = avg_pred
        test_dataset['yeer'] = yeer
        x = list(test_dataset['flashpoint'].values)
        y = list(test_dataset['pred'].values)
        # set figure parameters
        fg = seaborn.FacetGrid(data=test_dataset, hue='source', height = 8, aspect=1.25)
        fg.map(plt.errorbar,                  # type of plot
               'flashpoint', 'pred', 'yeer',  # data column
               fmt = 'o', markersize = 4      # args for errorbar
              ).add_legend()                  # add legend
        # set x,y limit
        min_val = min(min(y),min(y)-max(yeer),min(x)-max(yeer)) - 20
        max_val = max(max(y),max(y)+max(yeer),max(x)+max(yeer)) + 20
        x = min_val
        y = max_val
        if text != None:
            i =  (max_val-min_val)/20
            for key in sorted(text.keys()):
                if key.find('list') == -1:
                    if key == 'train':
                        t = str(key+': '+ str(text[key]))
                    else:
                        t = str(key +': ' + str(round(text[key],4)))                    
                    if key == 'AAD':
                        t = t+str('%')
                    plt.text(x,y - i,t)
                    i += (max_val-min_val)/15
        fg.set(xlim = (min_val,max_val), ylim =(min_val, max_val))
        for ax in fg.axes.flat:
            ax.plot((min_val, max_val),(min_val, max_val))
        plt.title("Parity Plot")
        plt.ylabel("Predicted")
        plt.xlabel("Experimental")
        seaborn.despine(fg.fig,top=False, right=False)#, left=True, bottom=True,)
        plt.savefig('./'+dir_name+'/'+plot_name+'.png', dpi = 500,  bbox_inches='tight')
        plt.clf()
        plt.close()

    def residual_histogram(pred, dataset, plot_name = 'histogram', dir_name = 'residual_plot',text = None):
        # Create target Directory if don't exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("||||||||||||||||Directory residual_plot Created||||||||||||||||")
        expt = dataset['flashpoint'].tolist()
        residual = []
        for i in range(len(dataset)):
            residual.append((expt[i] - pred[i]))
       # plt.style.use('ggplot')
        plt.rc('font', size = 14)                                # change plot font size
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.hist(residual, bins = 50)
        plt.title("Histogram of the Residuals", fontsize = 14)
        plt.ylabel("Frequency", fontsize = 14)
        plt.xlabel("Residual", fontsize = 14)
        left,right = plt.xlim()
        bottom,top = plt.ylim()
        i = top/20
        if text != None:
            for key in sorted(text.keys()):
                if key.find('list') == -1:
                    if key == 'train':
                        t = str(key+': '+ str(text[key]))
                    else:
                        t = str(key +': ' + str(round(text[key],4)))
                    if key == 'AAD':
                        t = t+str('%')
                    plt.text(left,top - i,t)
                    i += top/15
        plt.savefig('./'+dir_name+'/'+plot_name+'.png', dpi = 500, bbox_inches='tight')
        plt.clf()
        plt.close()

    def interactive_plot(pred_result,true_result):
        return 0 
    

class HyperparamOpt(object):
  """
  Provides simple hyperparameter search capabilities.
  """

  def __init__(self, model_class, verbose=True):
    self.model_class = model_class
    self.verbose = verbose

  def CVgridsearch (self,
                    params_dict,
                    dataset, #added by Sean
                    model = 'MPNN',
                    n_CV = 5, # added by Sean
                    use_max=False,
                    logdir=None):
    metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    hyperparams = params_dict.keys()
    hyperparam_vals = params_dict.values()
    for hyperparam_list in params_dict.values():
        assert isinstance(hyperparam_list, collections.Iterable)

    number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

    if use_max:
        best_validation_score = -np.inf
    else:
        best_validation_score = np.inf
    best_hyperparams = None
    best_model, best_model_dir = None, None
    all_scores = {}
    if model == 'MPNN':
        feat = dc.feat.WeaveFeaturizer()
    elif model == 'GCNN':
        feat = dc.feat.ConvMolFeaturizer()
    for ind, hyperparameter_tuple in enumerate(
        itertools.product(*hyperparam_vals)):
        valid_scores = []
        train_scores = []
        model_params = {}
        print("==========Fitting model %d/%d==========" % (ind + 1, number_combinations), self.verbose)
        for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
            model_params[hyperparam] = hyperparam_val
        log("hyperparameters: %s" % str(model_params), self.verbose)

        if logdir is not None:
            model_dir = os.path.join(logdir, str(ind))
            print("model_dir is %s" % model_dir, self.verbose)
            try:
                os.makedirs(model_dir)
            except OSError:
                if not os.path.isdir(model_dir):
                    print("Error creating model_dir, using tempfile directory",
                        self.verbose)
                model_dir = tempfile.mkdtemp()
        else:
            model_dir = tempfile.mkdtemp()

        indices,new_dataset = Splitter.k_fold(dataset,n_splits = n_CV,getinfo = False)
        i = 0
        for train,val in indices:
            print("\n----------Conducting CV "+ str(i)+"----------")
            train_dataset = new_dataset.iloc[train]
            valid_dataset = new_dataset.iloc[val]
            train_dataset.to_csv('train_set.csv',index = False)
            valid_dataset.to_csv('valid_set.csv',index = False)
            flashpoint_tasks = ['flashpoint']
            loader = dc.data.CSVLoader(tasks = flashpoint_tasks,
                                        smiles_field="smiles",
                                        featurizer = feat)
            train_dataset = loader.featurize("train_set.csv", shard_size=8192)
            valid_dataset = loader.featurize("valid_set.csv", shard_size=8192)
            transformers = [
                dc.trans.NormalizationTransformer(
                transform_y=True, dataset=train_dataset, move_mean=True) 
            ]
            for transformer in transformers:
                train_dataset = transformer.transform(train_dataset)
                valid_dataset = transformer.transform(valid_dataset)
            model = self.model_class(model_params, model_dir)
            model.fit(train_dataset)
            evaluator = Evaluator(model, valid_dataset, transformers)
            evaluator_train = Evaluator(model, train_dataset,transformers)
            multitask_scores = evaluator.compute_model_performance([metric])
            log("train score:",self.verbose)
            train_multitask_scores = evaluator_train.compute_model_performance([metric])
            valid_scores.append(multitask_scores[metric.name])
            train_scores.append(train_multitask_scores[metric.name])
            i += 1
            os.remove("train_set.csv")
            os.remove("valid_set.csv")
        valid_score = sum(valid_scores)/n_CV
        train_score = sum(train_scores)/n_CV
        all_scores[str(hyperparameter_tuple)] = valid_score
        if (use_max and valid_score >= best_validation_score) or (
              not use_max and valid_score <= best_validation_score):
            best_validation_score = valid_score
            best_hyperparams = hyperparameter_tuple
            if best_model_dir is not None:
                shutil.rmtree(best_model_dir)
            best_model_dir = model_dir
            best_model = model
        else:
            shutil.rmtree(model_dir)

        log(
           "Model %d/%d, Metric %s, Validation set %s: %f, Train set %s: %f" %
           (ind + 1, number_combinations, metric.name, ind, valid_score, ind, train_score),
           self.verbose)
        log("\tbest_validation_score so far: %f" % best_validation_score,
            self.verbose)
    log("||||||||||Best hyperparameters: %s" % str(best_hyperparams), self.verbose)
    log("||||||||||validation_score: %f" % best_validation_score, self.verbose)
    return best_model, best_hyperparams, all_scores
