#!/usr/bin/python3
"""
Script defining the visualizations functions.

We have chosen to not save the PCA and UMAP components as it is not necessary for any other analysis. As such, it is not optimized for runtime (for example, for plotting the side by side UMAPs in best_model.py, the UMAP are computed twice for no reason).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import pandas as pd
import seaborn as sns


def plot_PCA(X, y, title, plotted_components=(1,2), save_path_pca=None):
    n_components = max(plotted_components)
    
    pca = PCA(n_components)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    X_r = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    data_representation = pd.DataFrame(X_r)
    data_representation.columns = [f"PCA{n+1}" for n in range(n_components)]
    data_representation["Labels"] = y
    
    ax = sns.scatterplot(data_representation, x = f"PCA{plotted_components[0]}", y = f"PCA{plotted_components[1]}",
                    hue = "Labels", size = 3, alpha = 0.5)
    # Cosmetics
    plt.title(title, fontsize=14)

    ax.set_xlabel(f'PC{plotted_components[0]} ({explained_variance[plotted_components[0]-1]*100:.2f}%)', fontsize=10)
    ax.set_ylabel(f'PC{plotted_components[1]} ({explained_variance[plotted_components[1]-1]*100:.2f}%)', fontsize=10)
        
    # Save the plot if save_path is provided
    if save_path_pca:
        plt.savefig(save_path_pca, format = "svg")
        print(f"Plot saved as {save_path_pca}")
    
    return fig, ax

def get_ax_UMAP(X, y, seed, ax=None):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    umap_model = umap.UMAP(n_components=2, random_state = seed)
    X_r = umap_model.fit_transform(X_scaled)
    
    data_representation = pd.DataFrame(X_r)
    data_representation.columns = ["UMAP1", "UMAP2"]
    data_representation["Labels"] = y
    
    ax = sns.scatterplot(data_representation, x = "UMAP1", y = "UMAP2", 
                    hue = "Labels", size = 3, alpha = 0.5, ax = ax)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    
    return ax


def compare_performance(model_data, save_path = "../plots/compare_perf"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.barplot(data=model_data, x="Model", y="-10log10(1-roc_auc)")


    plt.xticks(rotation=45, ha='right', fontsize=10)


    plt.title("Model Performance Comparison", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("-10log10(1-roc_auc)", fontsize=12)

    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  
    if save_path:
        plt.savefig(save_path, format = "svg")
        print(f"Plot saved as {save_path}")
    
    plt.show()
    
