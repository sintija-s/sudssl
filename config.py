import openml

# Load the dataset suite from OpenML-CC18
cc18_suite = openml.study.get_suite("OpenML-CC18")
datasets_l = cc18_suite.data # List of dataset IDs in the CC18 suite

# Configuration of random seeds
random_seeds_l = [2, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Configurable list of labeled data amounts for semi-supervised learning
number_of_labeled_examples_l = [50, 100, 200, 350, 500]

# All experimental settings for running tassel.py are below:
MODELS=['tabnet', 'scarf']
DATASETS=[3, 6, 28, 32, 44, 46, 151, 182, 300, 1590, 4134, 1489, 1497, 1486, 1475, 4534, 1461, 4538, 1478, 40499, 40668, 40983, 41027, 23517, 40923, 40978, 40670, 40701]
SEEDS=[2, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_LABELED=[50, 100, 150, 200, 350, 500, 750, 1000]
SAMPLING_METHOD=['random', 'consensus_ds', 'consensus_dt', 'baseline_all', 'baseline_none']
N_UNLABELED=[100, 250, 500, 750, 1000]
BRACKETS =["[0, 40]", "[20, 60]", '[40, 80]', '[60, 100]']
