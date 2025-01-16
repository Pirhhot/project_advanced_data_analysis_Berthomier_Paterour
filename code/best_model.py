#!/usr/bin/python3
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.datasets import make_classification
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from model_selection import log_transform
from viz_raw_data import get_ax_UMAP

class BestModel:
    """
    This class is used for creating the models and methods used for visualisations of the results and feature importance analysis.
    """
    def __init__(self, model_class, best_params, threshold_var = 1, seed=42, oversampling = False):
        """
        Initialize the BestModel class with necessary attributes for training and evaluation.
        
        Parameters:
        - model_class: Scikit Learn model class of choice
        - best_params: Dictionary containing best parameters for the model
        - seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.best_params = best_params
        self.seed = seed
        self.model = None
        self.pipeline = None
        self.oversampling = oversampling
        self.var_threshold = VarianceThreshold(threshold=threshold_var)
        
    def train_model(self, X_train, y_train, class_weights=None):
        """
        Train the best model based on the provided best parameters.
        """
        log_transformer = FunctionTransformer(log_transform)
        standard_scaler = StandardScaler()
        smt = SMOTE(random_state = self.seed)
        
        if class_weights:
            try :
                self.model = self.model_class(**self.best_params, random_state = self.seed, probability = True, class_weight = class_weights)
            except TypeError: #because SVC needs probability but it raises an error for the other classes
                self.model = self.model_class(**self.best_params, random_state = self.seed, class_weight = class_weights)
        else :
            try :
                self.model = self.model_class(**self.best_params, random_state = self.seed, probability = True, class_weight = class_weights)
            except TypeError: 
                self.model = self.model_class(**self.best_params, random_state = self.seed, class_weight = class_weights)

        if self.oversampling:
            self.pipeline = make_pipeline(log_transformer, self.var_threshold, smt, standard_scaler, self.model)
        else : 
            self.pipeline = make_pipeline(log_transformer, self.var_threshold, standard_scaler, self.model)
        
        self.pipeline.fit(X_train, y_train)
        
    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, multi_class = "ovr", average = "weighted")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC Score (OVR Weighted) : {roc_auc_weighted:.3f}")
    
    def side_by_side_UMAP(self, X_test, y_test):

        fig_umap, axs_umap = plt.subplots(1, 2, figsize=(20, 10))
    
        get_ax_UMAP(X_test, y_test, seed=self.seed, ax = axs_umap[0])
        axs_umap[0].set_title("Target values")
    
        y_pred = self.pipeline.predict(X_test)
        get_ax_UMAP(X_test, y_pred, seed=self.seed, ax = axs_umap[1])
        axs_umap[1].set_title("Predicted values")
    
        plt.savefig("../plots/UMAP_transcriptome", format="svg")
        print("Plot saved as UMAP_transcriptome")
    
        return fig_umap, axs_umap
    
   
    def ROC_AUC_curve(self, X_test, y_test):

        # Get probabilities and binarize labels
        y_pred_proba = self.pipeline.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
        # Plot ROC curves
        n_classes = y_test_bin.shape[1]

        plt.figure(figsize=(10, 7))
        for i, diag in enumerate(np.unique(y_test)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {diag} vs Rest (AUC = {roc_auc:.2f})")

        # Plot formatting
        plt.plot([0, 1], [0, 1], "k--", lw=2) #Diagonal random line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)
        plt.savefig("../plots/ROC_AUC_curve", format = "svg")
        plt.show()
    
    def process_genes_coeff(self, data):
        df = pd.DataFrame(self.model.coef_.T)
        df.columns = self.model.classes_
        
        selected_indices = np.where(self.var_threshold.get_support())[0]
        selected_gene_names = data.df.columns[selected_indices]
        
        df["Genes"] = selected_gene_names
        df_melted = pd.melt(df, id_vars=['Genes'], value_vars=pd.Series(data.y).unique(),
                    var_name='diagnosis_category', value_name='coeff_values')
        df_melted["coeff squared"] = df_melted["coeff_values"]**2
        df_melted["sign"] = np.sign(df_melted["coeff_values"])
        
        self.genes_coeff =  df_melted
        
    def plot_top_genes(self, n_genes=10):
        
        df_top_genes = self.genes_coeff.groupby('diagnosis_category').apply(
            lambda x: x.nlargest(n_genes, 'coeff squared')
        ).reset_index(drop=True)

        # Create a catplot with different x axes for each diagnosis category
        g = sns.catplot(data=df_top_genes, kind="bar", x="Genes", y="coeff squared", hue="sign", 
                col="diagnosis_category", col_wrap=2, sharex=False)

        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        g.set_titles("{col_name}")
        plt.suptitle('Top Genes by Diagnosis Category', fontsize=16)
        plt.savefig("../plots/top_genes", format = "svg")
        plt.show()
        