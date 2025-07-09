# SUDSSL - Selecting Unlabeled Data for Tabular Self-Supervised Learning

SUDSSL is a framework for subsampling the unlabeled data for self-supervised learning with tabular data.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3122/)


## Overview

Our methodology examines whether selecting unlabeled examples can maintain the predictive performance of a
tabular self-supervised classifier while cutting down computation costs.

![Methodology Overview](images/methodology_embeddings.svg)

Built using Python 3.12.2.


## Additional Results

Additional results for each evaluation stream can be found in the corresponding Jupyter notebooks:
- [i. Downstream Performance](1_evaluation_downstream_performance.ipynb)
- [ii. Embedding](2_evaluation_embedding.ipynb)
- [iii. Time](3_evaluation_time.ipynb)
  
For a more detailed analysis of the results, we offer the raw results upon request (as they are quite large). Please contact us in this case: [sintija.stevanoska@ijs.si](mailto:sintija.stevanoska@ijs.si).


## Creating a virtual environment (Recommended):
```bash
# Navigate to the folder where the repository is downloaded
cd your_folder

# Verify that Python 3.12.x is installed
#  (if not: download and install from https://www.python.org/downloads/)
py --list

# Create the virtual environment
py -3.12 -m venv .venv

# Activating the virtual environment
#  For MacOS/Linux
source .venv/bin/activate
#  For Windows
.venv/Scripts/activate

# Install the dependencies
pip install -r requirements.txt

# [OPTIONAL] If you wish to use Jupyter Notebook, register the virtual environment
#  as a kernel for Jupyter Notebook (make sure to activate it in your notebook)
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "assl.venv"

# [OPTIONAL] Start up Jupyter Notebook
jupyter notebook
```

## Running an experiment

To run a configuration outlined in the paper, run `main.py`. 

Below is an example with all arguments:

```commandline
python main.py --model "tabnet" --dataset 3 --seed 2 --n_labeled 50 --sampling_method "consensus_dt" --bracket "[60,100]" --n_unlabeled 1000 
```

Arguments:
- `--model`: The self-supervised model that will be evaluated. Currently, the scirpt supports "tabnet" or "scarf".
- `--dataset`: The ID of the OpenML dataset to be used.
- `--seed`: Random seed (default set to 42).
- `--n_labeled`: Number of labeled examples to use.
- `--sampling_method`: The sampling method to use. Options: "random", "consensus_ds", "consensus_dt", "baseline_all", "baseline_none".
- `--n_unlabeled`: Number of unlabeled examples to select; used only when `--sampling_method` is "random", "consensus_ds", or "consensus_dt".
- `--bracket`: Sampling bracket, put it as `"[minimum_value, maximum_value]"`; used only when `--sampling_method` is either "consensus_ds", or "consensus_dt".

## Extracting metafeatures

To extract metafeatures for a dataset, run the `extract_metafeatures.py`.
It produces a .csv file which has the selected metafeatures for 3 random seeds (including the averaged values in the last line). 

Below is an example of how to run it:
```commandline
python extract_metafeatures.py --dataset "3,6" --metafeature_type "statistical, clustering"
```
Arguments:
- `--dataset`: The ID of the OpenML dataset for which the metafeatures are extracted.
- `--metafeature_type`: Metafeature type, as defined by the PyMFE library. Options: "clustering", "complexity", 
"concept", "general", "info-theory", "itemset", "landmarking", "model-based", and "statistical".

