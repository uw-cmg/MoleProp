# MoleProp
Code for paper: Assessing Graph-based Deep Learning Model for Predicting Flash Point

# Structure

## Data
Our data sets are shared on Figshare at 10.6084/m9.figshare.9275210

## Util
- integration tool: contains methods used to collect and clean data 
- workflow tool: Main file for opimization, training and testing

## data_engineering
Provides the classes and jupyter notebooks for building the integrated dataset, the Gelest datasets, the Pubchem dataset, and the notebooks used for web scraping and data cleaning

## gcnn_paper_comparison
GCNN optimization and testing files for paper comparisons

- optimization folder contains the optimization file we used to optimize GCNN models on each paper with nested 5-fold cross validation and the batch file we used to submit our jobs to UW-Madison' Euler computer cluster. It also contains training and test sets for each fold.
- test folder contains the test file we used for testing GCNN model on each paper and a batch file to submit our jobs. **Every test file shows the optimized hyperparameter we got from optimization for each paper**


## mpnn_paper_comparison
MPNN optimization and testing files for paper comparisons

- optimization folder contains the optimization file we used to optimize MPNN models on each paper with nested 5-fold cross validation and the batch file we used to submit our jobs to UW-Madison' Euler computer cluster. It also contains training and test sets for each fold.
- test folder contains the test file we used for testing MPNN model on each paper and a batch file to submit our jobs. **Every test file shows the optimized hyperparameter we got from optimization for each paper**

## entire_and_chemical_tests
Optimization and testing files for both full integrated dataset tests and chemistry subset tests

- optimization folder contains the script used to run the hyperparameter optimization as well as the required dataset file
- test folder contains the script and dataset files used to train and test the optimized model, note that the optimizal hyperparameters are embedded in each script and can thus be used for referencing if desired
