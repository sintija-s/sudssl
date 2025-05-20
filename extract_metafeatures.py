import argparse
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
import util
from dataset import load_cc18_dataset


def compute_metafeatures_for_dataset(X, y, random_state, categorical_column_names, metafeature_type):
    mf_names, mf_values = extract_metafeatures(
        X, y, random_state, categorical_column_names, metafeature_type
    )
    mf_df = pd.DataFrame(
        [mf_values], columns=[mf_names], index=[f"mf_{str(random_state)}"]
    )
    return mf_df


def extract_metafeatures(X, y, random_state, categorical_column_names, metafeature_type):

    results_names, results_values = [], []

    for group in metafeature_type:
        names, values = run_metafeature_extractor(
            random_state, group, X, y, categorical_column_names
        )
        results_names += names
        results_values += values

    return results_names, results_values


def run_metafeature_extractor(
    random_state, metafeature_group, X, y, categorical_column_names
):
    mfe = MFE(
        groups=(metafeature_group),
        summary=["max", "min", "mean"],
        random_state=random_state,
        score="f1",
    )
    has_only_categorical = len(X.columns) == len(categorical_column_names)
    if has_only_categorical:
        mfe.fit(
            X.values,
            y.values,
            cat_cols=categorical_column_names,
            transform_num=False,
            transform_cat="gray",
            suppress_warnings=True,
        )
    else:
        mfe.fit(
            X.values,
            y.values,
            cat_cols=categorical_column_names,
            transform_num=False,
            transform_cat=None,
            suppress_warnings=True,
        )
    return mfe.extract(
        suppress_warnings=True,
        cat_cols=categorical_column_names,
        transform_num=False,
        transform_cat="gray" if has_only_categorical else None,
    )


def read_and_preprocess_datasets(fname):
    x, y = load_cc18_dataset(fname)
    # identify categorical columns
    categorical_column_names = x.select_dtypes(
        include=["object", "category"]
    ).columns.to_list()
    categorical_col_indices = [
        x.columns.get_loc(col) for col in categorical_column_names
    ]
    numerical_column_names = x.drop(columns=categorical_column_names).columns.to_list()
    numerical_col_indices = [x.columns.get_loc(col) for col in numerical_column_names]
    unique_vals_within_category = []
    for col in categorical_column_names:
        unique_vals_within_category.append(len(x[col].unique()))

    label_encoder = LabelEncoder()

    x[categorical_column_names] = x[categorical_column_names].astype("object")
    # standardize numerical features
    if numerical_column_names:
        scaler = StandardScaler()
        x.loc[:, numerical_column_names] = scaler.fit_transform(
            x.loc[:, numerical_column_names]
        )
    y = pd.Series(label_encoder.fit_transform(y))

    return x, y, categorical_col_indices


def main(dataset=None, metafeature_type=None):
    if dataset:
        # Convert string representation to list of integers
        if isinstance(dataset, str):
            dataset_list = [int(ds.strip()) for ds in dataset.split(',') if ds.strip()]
        else:
            dataset_list = dataset
    else:
        exclude_datasets = [40927, 40996, 554]  # cifar10 fashionmnist, mnist
        DATASETS = [d for d in config.datasets_l if d not in exclude_datasets]
        dataset_list = DATASETS
        # dataset_list = config.fname_only_categorical_values

    # Handle metafeature types
    valid_metafeature_types = ['clustering', 'complexity', 'concept', 'general', 'info-theory', 'itemset', 'landmarking', 'model-based', 'statistical']
    
    if metafeature_type is None:
        metafeature_type = ['statistical']
    elif isinstance(metafeature_type, str):
        # Split by comma and strip whitespace and any square brackets
        metafeature_type = [mf.strip().strip('[]') for mf in metafeature_type.split(',') if mf.strip()]

    # Validate each metafeature type
    for mf_type in metafeature_type:
        if mf_type not in valid_metafeature_types:
            raise ValueError(f"Invalid metafeature type: {mf_type}. Please choose from: {valid_metafeature_types}")
    fname_appendix = '_'.join(metafeature_type)

    output_dir = os.path.join("results", "metafeatures")
    os.makedirs(output_dir, exist_ok=True)
    for fname in dataset_list:

        print(f"--------DATASET: {fname}--------")
        metafeatures = pd.DataFrame()
        x, y, categorical_column_names = read_and_preprocess_datasets(fname)
        for random_state in config.random_seeds_l[:3]:
            print(f"----SEED: {random_state}")
            util.set_seeds(random_state)
            metafeatures_df = compute_metafeatures_for_dataset(
                x, y, random_state, categorical_column_names, metafeature_type
            )
            metafeatures = pd.concat([metafeatures, metafeatures_df])
        metafeatures.loc[fname] = metafeatures.mean()

        output_path = os.path.join(output_dir, f"{fname}_metafeatures_{fname_appendix}.csv")
        metafeatures.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metafeatures from datasets.")

    # Add an optional argument for the dataset
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the dataset to process. If not provided, all datasets will be processed.",
    )
    parser.add_argument(
        "--metafeature_type",
        type=str,
        default="statistical",
        required=False,
        help="Comma-separated list of metafeature types to extract (no brackets needed). Choose from: clustering, complexity, concept, general, info-theory, itemset, landmarking, model-based, statistical",
    )

    args = parser.parse_args()
    main(dataset=args.dataset, metafeature_type=args.metafeature_type)
