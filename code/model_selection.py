#!/usr/bin/python3
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE 
import numpy as np

def log_transform(X):
    """
    Applies a log transformation to    the input data using the natural logarithm.
    The `np.log1p` function is numerically stable for small values and ensures the 
    transformation is defined even for zero values (log(1 + X)).

    Parameters:
    ----------
    X : array-like
        The input data to be transformed.

    Returns:
    -------
    np.ndarray
        The log-transformed data as a NumPy array.
        For each element x in X, the transformation applied is log(1 + x).
    """
    X = np.asarray(X)  
    X_transformed = np.log1p(X)  
    return X_transformed 

def nested_cross_val(data, model, param_grid, outer_cv, inner_cv, seed, threshold_var=1, oversampling = False, grid_search_score = "roc_auc_ovr"):
    """
    Performs nested cross-validation for hyperparameter tuning and model evaluation. 
    Take in data, model, param_grid and cross_validations objects and handle preprocessing and optionnal oversampling with SMOTE.

    Parameters:
    ----------
    data : object
        A data object containing `X_train` (features) and `y_train` (target labels).
        `X_train` and `y_train` should be NumPy arrays or array-like structures.

    model : sklearn.base.BaseEstimator
        The machine learning model to be trained and evaluated.

    param_grid : dict
        A dictionary defining the hyperparameter grid for GridSearchCV.

    outer_cv : sklearn.model_selection._BaseKFold
        Cross-validation splitting strategy for the outer loop (e.g., KFold, StratifiedKFold).

    inner_cv : sklearn.model_selection._BaseKFold
        Cross-validation splitting strategy for the inner loop used in GridSearchCV.

    seed : int
        Random seed for reproducibility in SMOTE and other stochastic processes.

    threshold_var : float, optional, default=1
        Threshold for feature selection using VarianceThreshold. Features with variance
        below this threshold are removed.

    oversampling : bool, optional, default=False
        Whether to apply SMOTE oversampling during training to handle class imbalance.

    grid_search_score : str, optional, default="roc_auc_ovr"
        The scoring metric for GridSearchCV (e.g., "roc_auc", "accuracy").

    Returns:
    -------
    dict
        A dictionary containing the best hyperparameters and scores for each outer fold:
        - "Params": List of best hyperparameters found in each fold.
        - "Score": List of model scores (ROC AUC) for each fold.
    """
    # Store the best parameters and scores for each outer fold
    best_params_per_fold = []
    best_score_per_fold = []
    variance_threshold = VarianceThreshold(threshold=threshold_var)
    log_transformer = FunctionTransformer(log_transform)
    standard_scaler = StandardScaler()
    gridsearch_inner= GridSearchCV(model, param_grid=param_grid, cv=inner_cv, n_jobs= -1, scoring=grid_search_score, verbose = 1)
    smt = SMOTE(random_state = seed)
    n = 1
    for train_index, test_index in outer_cv.split(data.X_train, data.y_train):
        print(f"Run {n}")
        
        # Split the data into training and test sets for this fold
        X_train_out, X_test_out = data.X_train[train_index], data.X_train[test_index]
        y_train_out, y_test_out = data.y_train[train_index], data.y_train[test_index]
        
        # Create a new pipeline to refit the gridsearch with an option for oversampling
        if oversampling:
            pipeline = make_pipeline(log_transformer, variance_threshold, smt, standard_scaler, gridsearch_inner)
        else : 
            pipeline = make_pipeline(log_transformer, variance_threshold, standard_scaler, gridsearch_inner)

        # Fit the pipeline on the training data of this fold
        pipeline.fit(X_train_out, y_train_out)
    
        # Get the best hyperparameters and corresponding scores found by GridSearchCV for this fold
        best_params = pipeline.named_steps['gridsearchcv'].best_params_
        y_pred_out = pipeline.predict_proba(X_test_out)
        best_score = roc_auc_score(y_test_out, y_pred_out, multi_class = "ovr", average = "weighted")
        best_params_per_fold.append(best_params)
        best_score_per_fold.append(best_score)
        
        n +=1
        
    # return the best hyperparameters found in each fold
    return {"Params": best_params_per_fold, 
            "Score": best_score_per_fold}


def cross_val_pipeline(pipeline, data, seed, cv): #just to lower the quantity of parameters to copy paste
    cv_scores_pipeline = cross_val_score(pipeline, data.X_train, data.y_train, cv=cv, scoring = "roc_auc_ovr_weighted", n_jobs = -1)
    return cv_scores_pipeline


def diffential_unbalanced_handling(model_unweighted, model_weighted, data, seed, cv):
    """
    Compares the performance of different strategies for handling unbalanced datasets using cross-validation.

    This function evaluates three approaches for handling class imbalance in classification problems:
    1. Using an unweighted model.
    2. Using a model with built-in class weights.
    3. Using SMOTE (Synthetic Minority Oversampling Technique) for oversampling.

    Parameters:
    ----------
    model_unweighted : sklearn.base.BaseEstimator
        The machine learning model without built-in class weight handling.

    model_weighted : sklearn.base.BaseEstimator
        The machine learning model configured to handle class weights (e.g., `class_weight='balanced'`).

    data : object
        A data object containing `X_train` (features) and `y_train` (target labels).
        `X_train` and `y_train` should be NumPy arrays or array-like structures.

    seed : int
        Random seed for reproducibility in SMOTE and other stochastic processes.

    cv : sklearn.model_selection._BaseKFold
        Cross-validation splitting strategy (e.g., KFold, StratifiedKFold).

    Returns:
    -------
    tuple
        A tuple of cross-validation scores for the three pipelines:
        - cv_scores_unweighted: Cross-validation scores for the unweighted model pipeline.
        - cv_scores_weighted: Cross-validation scores for the weighted model pipeline.
        - cv_scores_oversampled: Cross-validation scores for the SMOTE-oversampled model pipeline.
    """
    
    #creating the pipelines
    log_transformer = FunctionTransformer(log_transform)
    variance_threshold = VarianceThreshold(threshold=1)
    smt = SMOTE(random_state = seed)
    standard_scaler = StandardScaler()
    
    pipeline_unweighted = make_pipeline(log_transformer, variance_threshold, standard_scaler, model_unweighted)
    pipeline_weighted = make_pipeline(log_transformer, variance_threshold, standard_scaler, model_weighted)
    pipeline_oversampled = make_pipeline(log_transformer, variance_threshold, standard_scaler, smt, model_unweighted)
    
    #computing the cv scores
    cv_scores_unweighted = cross_val_pipeline(pipeline_unweighted, data, seed=seed, cv=cv)
    cv_scores_weighted = cross_val_pipeline(pipeline_weighted, data, seed=seed, cv=cv)
    cv_scores_oversampled = cross_val_pipeline(pipeline_oversampled, data, seed=seed, cv=cv)
    return cv_scores_unweighted, cv_scores_weighted, cv_scores_oversampled