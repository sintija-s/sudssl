import pandas as pd
import glob
import os

base_path='results/results/'
datasets = [1461,1475,1478,1486,1489,1497,151,1590,182,23517,28,300,32,3,40499,40668,40670,40701,
            40923,40978,40983,41027,4134,44,4534,4538,46,6]
all_dfs = []
for ds in datasets:
    all_ds_dfs = []
    for file_path in glob.glob(os.path.join(base_path, f'{ds}/*_tabnet.csv')):
        all_ds_dfs.append(pd.read_csv(file_path))
    all_ds_df = pd.concat(all_ds_dfs, ignore_index=True)
    all_ds_df.to_csv(f'results/per_dataset/{ds}_tabnet.csv', index=False)
    all_dfs.append(all_ds_df)
df = pd.concat(all_dfs, ignore_index=True)