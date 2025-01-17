Authors: Thomas Berthomier and Loann Paterour

Project: Transcriptome Analysis and classification of Bone Marrow neoplasia

This project is designed to analyze transcriptome data using both supervised and unsupervised learning techniques. It includes data preprocessing steps. The project then visualize the data using unsupervised methods such as PCA and UMAP. The project also incorporates machine learning pipelines for classification tasks (testing logistic regression, SVM and MLP) taking into account the unbalanced structure of the data.

This project provides a pipeline for analyzing transcriptomic datasets:

The implementation includes modular code with reusable classes and methods to ensure easy scalability and maintainability.

Preprocessing the data to have a final file with all the transcriptomes for all the patients and the labels asssociated.
Visualizing the data using PCA and UMAP for dimensionality reduction.
Building supervised machine learning models to classify samples, handling imbalanced datasets with computed class weights.
Analysis of the performance of the model.
Feature Importance analysis
Data Loading and Preprocessing:

Automatically loads transcriptomic data and clinical annotations. Filters unknown categories and handles class imbalance with computed weights.

Visualization:

Generates PCA plots to explain variance in the dataset. Generates UMAP plots for clustering and dimensionality reduction.

Machine Learning Pipelines:

Prebuilt pipelines for training and testing models (e.g., SVM, MLP). Implements cross-validation and nested cross-validation for model evaluation.

File Structure

project/
├── data/                        # Directory for data files
│   ├── merged_data_final.csv    # This file is too big to be put on this repository => Get it from Thomas or download the raw_counts from the id_summary.tsv => GDC database and use Data load method.
│   ├── clinical.tsv             # For preprocessing (merging)
│   ├── raw_count/               # Raw count data
│   └── id_summary.tsv           # For preprocessing (merging)
├── plots/                       # Directory for saved plots
├── output/                      # Directory for output files (e.g., model results, predictions)
├── code/                        # Directory for saved code
│   ├── datasets.py              # Data loading and preprocessing class of the main data
│   ├── viz_raw_data.py          # Functions for PCA and UMAP plotting
│   ├── best_model.py            # Functions for model selection
│   ├── model_selection.py            # Functions for the best model training and testing
│   ├── Project_visualization.ipynb   # Project markdown
│   └── models.py                # Main script to run the different models (svm, mlp)

└── README.md                    # Project documentation


Dependencies

The following Python libraries are required:

os pandas numpy matplotlib seaborn scikit-learn umap-learn imblearn json time

Data

The project expects two main data files:

merged_data_final.csv: Contains the main dataset with transcriptomic information and the labels. clinical.tsv: Contains clinical annotations with Case ID and diagnosis_category.

Place these files in the data/ directory.

Requirements

models.py has been launched using 60 CPUs and 256 Go of RAM. Run time with parallelisation : around (see /output/timer)
