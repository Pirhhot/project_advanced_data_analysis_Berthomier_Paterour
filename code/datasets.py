#!/usr/bin/python3
"""
Dataset script for importing and preprocess the data
"""
import json
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from viz_raw_data import get_ax_UMAP, plot_PCA

class Data():
    """ This class is created to store all of the data and the methods to load and transform it. """
    def __init__(self, data_path = "../data/", seed : int = 123, test = False, use_as_merger=False):
        """
    Initializes an object and loads data from specified paths. If the necessary files are found, 
    it processes and prepares data for machine learning. 
    Args:
        data_path (str): The path to the data directory. Default is "../data/".
        seed (int): The random seed used for data splitting. Default is 123.
        test (bool): A flag to determine whether to load a subset (500 rows) of the merged data for testing purposes.
                     Default is False (loads the full dataset).
        use_as_merger (bool): A flag indicating whether to skip data loading and processing for use as a merger. 
                               Default is False.

    Attributes:
        data_path (str): The data directory path where the files are located.
        df (pandas.DataFrame): A dataframe containing merged data from the "merged_data_final.csv" file.
        clinical_data (pandas.DataFrame): A dataframe containing clinical data from the "clinical.tsv" file.
        unknown (pandas.DataFrame): A subset of the data where the diagnosis category is "unknown".
        X (numpy.ndarray): Feature matrix (all columns except "diagnosis_category" and "Case ID").
        y (numpy.ndarray): Target vector (encoded "diagnosis_category").
        class_weights_dict (dict): A dictionary mapping class labels to computed class weights, 
                                   accounting for class imbalance.
        X_train (numpy.ndarray): Training feature matrix.
        X_test (numpy.ndarray): Test feature matrix.
        y_train (numpy.ndarray): Training target vector.
        y_test (numpy.ndarray): Test target vector.

    Process:
        - If the necessary data files (`merged_data_final.csv` and `clinical.tsv`) exist, the constructor:
            1. Loads the merged data and clinical data.
            2. Merges the data based on the "Case ID" column.
            3. Splits the data into features (`X`) and target (`y`), encoding the target labels using `LabelEncoder`.
            4. Computes class weights to address class imbalance using `compute_class_weight`.
            5. Splits the dataset into training and test sets using `train_test_split`.
        - If `use_as_merger` is set to `True`, no data loading or processing occurs, it is created in order to use the load_individual_RNAseq method.
        - If the required files are missing, an error message is printed indicating the issue.

    Side effects:
        - Loads and processes data files from the specified `data_path`.
        - Performs data cleaning, merging, and encoding operations.
        - Initializes attributes that hold the processed data and splits.
    """
        self.data_path = data_path
        
        if os.path.isfile(f"{data_path}merged_data_final.csv") and os.path.isfile(f"{data_path}clinical.tsv") and not use_as_merger:
            
            print("Loading the merged dataframe...")
            if test:
                self.df = pd.read_csv(f"{self.data_path}merged_data_final.csv", header = 0, nrows = 500)
            else :
                self.df = pd.read_csv(f"{self.data_path}merged_data_final.csv", header = 0)
            print("Done!")
            
            print("Loading clinical data...")
            self.clinical_data = pd.read_csv(f"{self.data_path}/clinical.tsv", sep="\t", header=0)
            print("Done!")
            
            print("Adding categories to the dataframe...")
            self.df = self.df.merge(self.clinical_data[["Case ID", "diagnosis_category"]].drop_duplicates(), on="Case ID", how="left")
            
            #Separating the unknown category from the rest
            self.unknown = self.df[self.df["diagnosis_category"] == "unknown"]
            self.df = self.df[self.df["diagnosis_category"] != "unknown"]
            
            print("Creating attributes for scikit-learn...")
            #X and y, we need to convert y to ints for certain functions like roc_auc_score
            self.X = self.df.drop(['diagnosis_category','Case ID'], axis=1, inplace=False).values
            self.y = self.df["diagnosis_category"].values
            
            #class weigth to take into account class imbalance
            class_weights = compute_class_weight('balanced', classes=self.df["diagnosis_category"].unique(), y=self.df["diagnosis_category"])
            self.class_weights_dict = dict(zip(
                self.df["diagnosis_category"].unique(),
                class_weights
            ))
            
            #Test and train
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.15, shuffle=True, stratify = self.y, random_state = seed)
            
            print("Data loading completed!")
            
        elif use_as_merger : #skips the loading 
            pass
        else:
            print("A file is missing or the data path is wrong.")
    
    def load_individual_RNAseq(self):
        """
        Load iteratively the individual RNA-seq files in the corresponding directory and add the data to the df attribute. 
        """
        id_filename = pd.read_csv(f'{self.data_path}id_summary.csv', header = 0)
        combined_RNAseq = pd.DataFrame()
        n_lines = len(id_filename)
        error_list = []
        for i, file_path in enumerate(id_filename["File Name"]) :
            try :
                unique_RNAseq = pd.read_csv(f"{self.data_path}raw_count/{file_path}", sep="\t", skiprows=[0,2,3,4,5], header=0)
                unique_RNAseq = unique_RNAseq.set_index("gene_name")["tpm_unstranded"].to_frame().T
                unique_RNAseq["Case ID"]= id_filename.loc[i,"Case ID"]
                combined_RNAseq = pd.concat([combined_RNAseq, unique_RNAseq])
            except :
                error_list.append(id_filename.loc[i,"Case ID"])
            if i % 15 == 0:
                print(f"Merging of the RNAseq_files : [{i}/{n_lines}]")
        self.df = combined_RNAseq
        print("Merging finished !")
        print(f"Merging errors : {error_list}")
        
    def plot_PCA(self, plotted_components=(1,2)):
        fig_pca, axs_pca = plot_PCA(self.X, self.y, "PCA of the transcriptomes", plotted_components=plotted_components, save_path_pca="../plots/PCA_transcriptome")
        return fig_pca, axs_pca
    
    def plot_UMAP(self, seed):
        fig_umap, axs_umap = plt.subplots(figsize=(10, 10))
        axs_umap = get_ax_UMAP(self.X, self.y, seed=seed)
        plt.title("UMAP of the transcriptomes", fontsize=14)
        
        plt.savefig("../plots/UMAP_transcriptome", format = "svg")
        print(f"Plot saved as UMAP_transcriptome")
    
        return fig_umap, axs_umap


def flatten(xss):
    return [x for xs in xss for x in xs]

def load_data_comparison(path = "../output"):
    list_scores = [
    "logistic_unweighted",
    "logistic_weighted",
    "logistic_oversampled",
    "svm_unweighted",
    "svm_weighted",
    "svm_oversampled",
    ]
    dir_scores = ["mlp_not_oversampled","mlp_oversampled"]


    tot_scores = []
    tot_cat = []
    tot_params = []
    for scores in list_scores :
        with open(f"../output/{scores}", "r") as file:
            loaded_list = json.load(file)
            tot_scores.append(loaded_list)
            tot_cat.append([scores]*len(loaded_list))
            tot_params.append(["base"]*len(loaded_list))

    for dirs in dir_scores :
        with open(f"../output/{dirs}", "r") as file:
            loaded_dict = json.load(file)
            tot_scores.append(loaded_dict["Score"])
            tot_cat.append([dirs]*len(loaded_dict["Score"]))
            tot_params.append(loaded_dict["Params"])
        
    out = pd.DataFrame({
        "Model" : flatten(tot_cat),
        "ROC-AUC" : flatten(tot_scores),
        "Params": flatten(tot_params)
    })
    out["-10log10(1-roc_auc)"] = -np.log10(1 - out["ROC-AUC"])
    return out