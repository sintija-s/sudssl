import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import set_config


# reads all csv files in one folder
# returns one concatenated DataFrame with all results for the sampling methods, and
#  a separate DataFrame with the baseline results
def read_and_concat_results_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter for the rows with the 'BaselineAll' and 'BaselineNone' sampling_method values
    baseline_df = df[df['sampling_method'].isin(['BaselineAll', 'BaselineNone'])]
    averaged_baseline_df = baseline_df.groupby(['dataset', 'n_labeled', 'sampling_method', 'metric'], as_index=False)['value'].mean()
    
    df_filtered = df[~df['sampling_method'].isin(['BaselineAll', 'BaselineNone'])]
    
    return df_filtered, averaged_baseline_df


def set_seeds(seed=42):
    # Set a random seed for NumPy
    np.random.seed(seed)
    random.seed(seed)
    # Define torch generator to preserve reproducibility in dataloaders
    g = torch.Generator()
    # Set a random seed for Scikit-learn
    set_config(assume_finite=True, print_changed_only=False)
    # Set a random seed for PyTorch Lightning
    # pl.seed_everything(seed)