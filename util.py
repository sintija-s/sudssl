import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import set_config

import glob
from matplotlib import pyplot as plt
import seaborn as sns


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif'],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.title_fontsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
    "figure.figsize": (9, 3)  # standard size for subplots
})
COLORS = ['#4c90b8', '#2ac3c1', '#f5b811', '#de653e', '#ff912a']
color_dict = {
    'BaselineAll': 'black',
    'BaselineNone': 'gray',
    'DisagreementSampling(DT)': COLORS[3],
    'DisagreementSampling(DS)': COLORS[4],
    'RandomSampling': COLORS[0],
    # add other sampling methods as needed
}


# reads all csv files in one folder
# returns one concatenated DataFrame with all results for the sampling methods, and
#  a separate DataFrame with the baseline results
def read_and_concat_results_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Filter for the rows with the 'BaselineAll' and 'BaselineNone' sampling_method values
    baseline_df = df[df["sampling_method"].isin(["BaselineAll", "BaselineNone"])]
    averaged_baseline_df = baseline_df.groupby(
        ["dataset", "n_labeled", "sampling_method", "metric"], as_index=False
    )["value"].mean()

    df_filtered = df[~df["sampling_method"].isin(["BaselineAll", "BaselineNone"])]

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


def get_all_linear_evaluation_results(base_results_folder):
    all_clustering_dfs = []
    all_linear_probing_dfs = []
    clustering_metrics = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ]
    linear_probing_metrics = [
        "auprc_macro_test",
        "accuracy_test",
        "balanced_accuracy_test",
        "f1_score_test",
    ]

    # for dataset_folder in os.listdir(base_results_folder):
    #     dataset_path = os.path.join(base_results_folder, dataset_folder)
    #     if os.path.isdir(dataset_path):
    #         pretraining_eval_path = os.path.join(dataset_path, "pretraining_evaluation")
    #         if os.path.isdir(pretraining_eval_path):
    #             # Find all CSV files matching the pattern
    #             search_pattern = os.path.join(
    #                 pretraining_eval_path, "linear_probe_results_*.csv"
    #             )
    #             for csv_file_path in glob.glob(search_pattern):
    #                 try:
    #                     df_current_file = pd.read_csv(csv_file_path)
    #                     model = csv_file_path.rsplit("_", 1)[-1].rsplit(".", 1)[0]
    #                     df_current_file["model"] = model
    #                     # Filter out rows with "_validation" in the metric column
    #                     df_current_filtered = df_current_file[
    #                         ~df_current_file["metric"].str.contains("_validation")
    #                     ]
    #                     # Create clustering results table for the current file
    #                     current_clustering = df_current_filtered[
    #                         df_current_filtered["metric"].isin(clustering_metrics)
    #                     ]
    #                     all_clustering_dfs.append(current_clustering)
    #                     # Create linear probing results table for the current file
    #                     current_linear_probing = df_current_filtered[
    #                         df_current_filtered["metric"].isin(linear_probing_metrics)
    #                     ]
    #                     all_linear_probing_dfs.append(current_linear_probing)
    #                 except Exception as e:
    #                     print(f"Error processing file {csv_file_path}: {e}")
    #
    # final_clustering_df = pd.DataFrame()
    # if all_clustering_dfs:
    #     final_clustering_df = pd.concat(all_clustering_dfs, ignore_index=True)
    #
    # final_linear_probing_df = pd.DataFrame()
    # if all_linear_probing_dfs:
    #     final_linear_probing_df = pd.concat(all_linear_probing_dfs, ignore_index=True)

    dfs = []
    for file_path in glob.glob(f'{base_results_folder}/results_embedding_*.csv'):
        dfs.append(pd.read_csv(file_path))
    df_all = pd.concat(dfs, ignore_index=True)

    clustering_df = df_all[df_all["metric"].isin(clustering_metrics)]
    linear_probing_df = df_all[df_all["metric"].isin(linear_probing_metrics)]

    return clustering_df, linear_probing_df


def clean_linear_probing_results(final_clustering_df, final_linear_probing_df):

    clustering_df = final_clustering_df.copy()
    linear_probing_df = final_linear_probing_df.copy()

    linear_probing_df["sampling_method"] = linear_probing_df["sampling_method"].replace(
        {
            "ConsensusSampling(DT)": "DisagreementSampling(DT)",
            "ConsensusSampling(DS)": "DisagreementSampling(DS)",
        }
    )

    bracket_cleaner = {
        40: "0-40",
        60: "20-60",
        80: "40-80",
        100: "60-100",
        0: "all",
        np.nan: "all",
    }

    linear_probing_df["bracket"] = linear_probing_df["bracket_max"].map(bracket_cleaner)
    clustering_df["bracket"] = clustering_df["bracket_max"].map(bracket_cleaner)

    linear_probing_df = linear_probing_df.drop(["bracket_min", "bracket_max"], axis=1)
    clustering_df = clustering_df.drop(["bracket_min", "bracket_max"], axis=1)

    return clustering_df, linear_probing_df


def pivot_linear_probing_results(linear_probing_df, join_keys):
    # For raw_features baseline
    final_linear_probing_df = linear_probing_df.copy()

    baseline_raw_features = final_linear_probing_df[
        final_linear_probing_df["trained_on"] == "raw_features"
    ].drop(
        ["bracket", "sampling_method", "n_sample", "actually_sampled", "trained_on"],
        axis=1,
    )
    baseline_raw_features = baseline_raw_features.rename(
        columns={"value": "baseline_raw_features"}
    )
    baseline_raw_features = baseline_raw_features.drop_duplicates(subset=join_keys)

    baseline_raw_features.drop_duplicates(inplace=True)

    # For BaselineNone
    baseline_none = final_linear_probing_df[
        (final_linear_probing_df["trained_on"] == "embedding")
        & (final_linear_probing_df["sampling_method"] == "BaselineNone")
    ].drop(
        ["bracket", "sampling_method", "n_sample", "actually_sampled", "trained_on"],
        axis=1,
    )
    baseline_none = baseline_none.rename(columns={"value": "baseline_none"})
    baseline_none = baseline_none.drop_duplicates(subset=join_keys)

    # For BaselineAll
    baseline_all = final_linear_probing_df[
        (final_linear_probing_df["trained_on"] == "embedding")
        & (final_linear_probing_df["sampling_method"] == "BaselineAll")
    ].drop(
        ["bracket", "sampling_method", "n_sample", "actually_sampled", "trained_on"],
        axis=1,
    )
    baseline_all = baseline_all.rename(columns={"value": "baseline_all"})
    baseline_all = baseline_all.drop_duplicates(subset=join_keys)

    sampled = final_linear_probing_df[
        (final_linear_probing_df["trained_on"] == "embedding")
        & (
            ~final_linear_probing_df["sampling_method"].isin(
                ["BaselineNone", "BaselineAll"]
            )
        )
    ]

    what = pd.merge(sampled, baseline_none, on=join_keys, how="left")
    what = pd.merge(what, baseline_all, on=join_keys, how="left")
    what = pd.merge(what, baseline_raw_features, on=join_keys, how="left")
    what.drop(["actually_sampled", "trained_on"], inplace=True, axis=1)

    return what


def get_best_performing_per_metric(df, aggregation_group, metric="auprc_macro_test"):
    df_per_metric = df[df["metric"] == metric]
    idx_max_value = df_per_metric.groupby(aggregation_group)["value"].idxmax()
    df_best_bracket = df_per_metric.loc[idx_max_value]

    return df_best_bracket

def make_baselines_for_q1(use_model, df):
    if use_model == "both":
        q1_df = df.copy()
    else:
        q1_df = df.loc[df['model'] == use_model].copy()

    # calculate percentages
    q1_df['of_baseline_all'] = (q1_df['value'] / q1_df['baseline_all']) * 100
    q1_df['of_baseline_none'] = (q1_df['value'] / q1_df['baseline_none']) * 100
    q1_df['of_baseline_raw'] = (q1_df['value'] / q1_df['baseline_raw_features']) * 100
    q1_df['none_of_all'] = (q1_df['baseline_none'] / q1_df['baseline_all']) * 100

    return q1_df

def get_pivot_table_for_q1(df, value_columns, aggregation='mean'):
    df_prepivot = df.copy()
    # Build aggfunc dict dynamically: mean for each value column
    aggfunc = {col: aggregation for col in value_columns}

    pivot_summary = pd.pivot_table(
        df_prepivot,
        index=["n_labeled", "sampling_method"],
        values=value_columns,
        aggfunc=aggfunc,
    )

    # Add pretty-printed columns for each value column
    for col in value_columns:
        pretty_col = f"% {col.replace('_', ' ').title()}"
        pivot_summary[pretty_col] = pivot_summary.apply(
            lambda row: f"{int(row[col])}", axis=1
        )

    pivot_summary = pivot_summary.drop(columns=value_columns)

    pivot_summary.rename(columns={"dataset": "Total Datasets"}, inplace=True)

    return pivot_summary

def get_plot_q1(use_model, df):
    if use_model == "both":
        df_tmp = df.copy()
    else:
        df_tmp = df.loc[df['model'] == use_model].copy()

    fig, ax = plt.subplots(figsize=(6, 2))  # Set width and height in inches

    g = sns.lineplot(
        data=df_tmp,
        x='n_sample',
        y='% of BaselineAll',
        hue='sampling_method',
        palette=color_dict,
        errorbar=None,
        ax=ax
    )

    ax.set_xlabel('n_sample')
    ax.set_ylabel('AUPRC\n% of BaselineAll')

    # Move legend to the right outside the plot
    legend = ax.legend(
        title='Sampling Method',
        bbox_to_anchor=(1.1, 0.5),
        loc='center left',
        borderaxespad=0,
        frameon=False
    )

    plt.tight_layout()
    # plt.savefig('plots/1_2_nsample_embedding_space.pdf', bbox_inches='tight')
    plt.show()
