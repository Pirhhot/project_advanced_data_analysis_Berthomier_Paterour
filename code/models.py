#!/usr/bin/python3

import os 
print(os.getcwd())

#import 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import time

from datasets import Data
from model_selection import nested_cross_val, diffential_unbalanced_handling
seed = 123


# DATA
data = Data()

# initialize crossvalidation
cv = KFold(n_splits=5, shuffle=True, random_state=seed) # Set a 5-fold CV

inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed+1)

#prepocessing initialization

# Models initialization
logistic_unweighted = LogisticRegression(penalty="l2", solver="liblinear")
logistic_weighted = LogisticRegression(penalty="l2", solver="liblinear", class_weight=data.class_weights_dict)
svm_unweighted = SVC(kernel ='linear', probability=True, C = 1, random_state = seed)  
svm_weighted = SVC(kernel ='linear', probability=True, class_weight = data.class_weights_dict, C = 1, random_state = seed)
mlp = MLPClassifier(random_state = seed) 

#hyperparameters tested :
param_grid_mlp = {
    'hidden_layer_sizes': [(500,), (1000,), (1000,500)],  
    'activation': ['relu', 'logistic'],                        
    'learning_rate_init': [0.001, 0.01],               
    'batch_size': [32]                              
}

#initialize cv objects 
outer_cv = KFold(n_splits = 5, shuffle = True, random_state=seed) 
inner_cv = KFold(n_splits = 5, shuffle = True, random_state=seed+1)

#run models

##logistic

print("Training Logistic...")

time0 = time.time()

cv_scores_logistic_unweighted, cv_scores_logistic_weighted, cv_scores_logistic_oversampled = diffential_unbalanced_handling(logistic_unweighted, logistic_weighted, data, seed, cv = outer_cv)

time_logistic = time.time() - time0
##SVM

print("Training SVM...")

time0 = time.time()

cv_scores_svm_unweighted, cv_scores_svm_weighted, cv_scores_svm_oversampled = diffential_unbalanced_handling(svm_unweighted, svm_weighted, data, seed, cv = outer_cv)

time_SMV = time.time() - time0

##MLP

print("Training MLP...")

time0 = time.time()

mlp_scores_not_oversampled = nested_cross_val(data, mlp, param_grid_mlp, outer_cv, inner_cv, seed)
mlp_scores_oversampled = nested_cross_val(data, mlp, param_grid_mlp, outer_cv, inner_cv, seed)

time_MLP = time.time() - time0

print("Training finished!")

#store data

print("Storing data...")

timer = {"Logistic": time_logistic,
         "SVM": time_SMV,
         "MLP": time_MLP}

scores_dict = {
    "logistic_unweighted": list(cv_scores_logistic_unweighted),
    "logistic_weighted": list(cv_scores_logistic_weighted),
    "logistic_oversampled": list(cv_scores_logistic_oversampled),
    "svm_unweighted": list(cv_scores_svm_unweighted),
    "svm_weighted": list(cv_scores_svm_weighted),
    "vm_oversampled": list(cv_scores_svm_oversampled),
    "mlp_not_oversampled": mlp_scores_not_oversampled,
    "mlp_oversampled": mlp_scores_oversampled,
    "timer" : timer
}
for name, var in scores_dict.items():
    with open(f"../output/{name}", "w") as file :
        json.dump(var, file)

