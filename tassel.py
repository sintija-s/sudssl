import argparse
import contextlib
import csv
import os
import warnings

import numpy as np
from dataset import (
    load_data,
    labeled_unlabeled_split,
    create_performance_dict,
    balanced_train_val_split,
)
from sklearn.model_selection import train_test_split

import models
import sampling

warnings.filterwarnings("ignore")


def validate_arguments(args):

    # Validate sampling method and parameters
    methods_requiring_unlabeled = ["random", "consensus_ds", "consensus_dt"]

    # Validate n_unlabeled
    if args.sampling_method in methods_requiring_unlabeled:
        if args.n_unlabeled is None:
            parser.error(
                f"--n_unlabeled is required when using sampling method '{args.sampling_method}'"
            )
        if args.n_unlabeled <= 0:
            parser.error(
                f"--n_unlabeled must be greater than 0 for sampling method '{args.sampling_method}'"
            )

    # Validate that bracket is provided for consensus sampling methods
    if args.sampling_method.startswith("consensus") and not args.bracket:
        parser.error("--bracket is required when using consensus sampling methods!")

    if args.model not in ["tabnet", "scarf"]:
        parser.error("--model can be either 'tabnet' or 'scarf'!")

    # Parse bracket if provided
    bracket = None
    if args.bracket:
        # Remove brackets and split by comma
        bracket_str = args.bracket.strip("[]")
        try:
            bracket = [int(x.strip()) for x in bracket_str.split(",")]
            if len(bracket) != 2:
                raise ValueError("Bracket must have exactly 2 values [min,max]")
        except ValueError as e:
            parser.error(
                f"Error parsing bracket: {e}. Format should be: [min,max] e.g. [0,40]"
            )
        args.bracket = bracket


def run_single_exp(
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    seed,
    n_labeled,
    n_sample,
    sampling_method,
    sampling_strategy,
    categorical_column_idx,
    unique_vals_within_category,
    dataset_to_process,
    bracket=None,
    mean=None,
):

    single_results = []
    sampling = sampling_method()

    # split into labeled and unlabeled data (randomly)
    x_train_lab, x_train_unlab, y_train_lab, y_train_unlab = labeled_unlabeled_split(
        x_train, y_train, n_labeled, seed
    )
    y_train_unlab = [-1] * len(
        y_train_unlab
    )  # indicator for 'unlabeled'; no true labels

    # selection of sample of unlabeled data that will be taken into account in SSL
    x_train_sel, y_train_sel, std_dev, sampling_time = sampling.sample(
        x_train_lab,
        x_train_unlab,
        y_train_lab,
        n_sample,
        seed,
        sampling_strategy=sampling_strategy,
        bracket=bracket,
        mean=mean,
    )

    x_train_lab, x_val, y_train_lab, y_val = balanced_train_val_split(
        x_train_lab, y_train_lab, test_size=0.2, random_state=42
    )

    # Train the model and evaluate metrics
    with contextlib.redirect_stdout(None):
        if model == "tabnet":
            y_pred, y_pred_proba, model_time = models.run_selfsl_tabnet(
                x_train_lab,
                x_train_sel,
                y_train_lab,
                y_train_sel,
                x_val,
                y_val,
                x_test,
                y_test,
                n_labeled,
                n_sample,
                categorical_column_idx,
                unique_vals_within_category,
                seed,
                bracket,
                dataset_to_process,
                sampling_method=str(sampling),
            )

        elif model == "scarf":
            y_pred, y_pred_proba, model_time = models.run_scarf(
                x_train_lab,
                x_train_sel,
                y_train_lab,
                y_train_sel,
                x_val,
                y_val,
                x_test,
                y_test,
                n_labeled,
                n_sample,
                categorical_column_idx,
                unique_vals_within_category,
                seed,
                bracket,
                dataset_to_process,
                sampling_method=str(sampling),
            )

    performance_dict = create_performance_dict(
        y_test, y_pred, y_pred_proba, std_dev, sampling_time, model_time
    )

    # Append results to the list
    br = bracket if bracket else [None, None]
    for metric, value in performance_dict.items():
        single_results.append(
            [
                dataset_to_process,
                seed,
                n_labeled,
                n_sample,
                len(x_train_sel),
                str(sampling),
                sampling_strategy,
                br[0],
                br[1],
                mean,
                metric,
                value,
            ]
        )
    return single_results


def main(
    model, dataset_to_process, seed, n_labeled, sampling_method, n_unlabeled, bracket
):
    sampling_method_map = {
        "random": sampling.RandomSampling,
        "consensus_ds": lambda: sampling.ConsensusSampling(clf_base="DS"),
        "consensus_dt": lambda: sampling.ConsensusSampling(clf_base="DT"),
        "baseline_all": sampling.BaselineAll,
        "baseline_none": sampling.BaselineNone,
    }
    # Check if the provided sampling method is valid
    if sampling_method not in sampling_method_map:
        valid_methods = list(sampling_method_map.keys())
        raise ValueError(
            f"Invalid sampling method: {sampling_method}. Valid options are: {valid_methods}"
        )

    # Get the actual sampling method implementation
    method_impl = sampling_method_map[sampling_method]

    # Load data
    (
        x,
        y,
        categorical_columns,
        categorical_col_indices,
        numerical_columns,
        numerical_col_indices,
        unique_vals_within_category,
    ) = load_data(dataset_to_process)
    if x.shape[0] < 2813 or np.isnan(x).any():
        print(
            f"Dataset {dataset_to_process} has insufficient examples (minimum 2813) or nan values. Skipping..."
        )
        return

    # Setup paths
    result_dir = os.path.join(f"results/{str(dataset_to_process)}")
    os.makedirs(result_dir, exist_ok=True)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, random_state=seed
    )
    if len(x_train) - n_labeled <= 2.5 * n_unlabeled:
        print("Not enough unlabeled data. Skipping...")
        return

    results = []

    # Set appropriate parameters based on sampling method
    if method_impl in [sampling.BaselineAll, sampling.BaselineNone]:
        results += run_single_exp(
            model,
            x_train,
            x_test,
            y_train,
            y_test,
            seed,
            n_labeled,
            0,
            method_impl,
            None,
            categorical_col_indices,
            unique_vals_within_category,
            dataset_to_process=dataset_to_process,
        )
        file_path = os.path.join(
            result_dir, f"{seed}_{n_labeled}_{sampling_method}_{model}.csv"
        )

    elif method_impl == sampling.RandomSampling:
        # Random sampling doesn't need sampling strategy or bracket
        results += run_single_exp(
            model,
            x_train,
            x_test,
            y_train,
            y_test,
            seed,
            n_labeled,
            n_unlabeled,
            method_impl,
            None,
            categorical_col_indices,
            unique_vals_within_category,
            dataset_to_process=dataset_to_process,
        )

        file_path = os.path.join(
            result_dir,
            f"{seed}_{n_labeled}_{sampling_method}_{n_unlabeled}_{model}.csv",
        )

    else:
        # Consensus sampling needs sampling strategy and bracket
        results += run_single_exp(
            model,
            x_train,
            x_test,
            y_train,
            y_test,
            seed,
            n_labeled,
            n_unlabeled,
            method_impl,
            "percentile",
            categorical_col_indices,
            unique_vals_within_category,
            bracket=bracket,
            dataset_to_process=dataset_to_process,
        )

        file_path = os.path.join(
            result_dir,
            f"{seed}_{n_labeled}_{sampling_method}_{n_unlabeled}_{bracket}_{model}.csv",
        )

    # Write results
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "seed",
                "n_labeled",
                "n_sample",
                "n_selected",
                "sampling_method",
                "sampling_strategy",
                "bracket_min",
                "bracket_max",
                "mean",
                "metric",
                "value",
            ]
        )
        writer.writerows(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TASSEL")

    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=True,
        help="The model to be used for SelfSupervised Learning. Options: 'tabnet', 'scarf",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        required=True,
        help="The OpenML ID of the dataset to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for experiments; defaults to 42",
    )
    parser.add_argument(
        "--n_labeled",
        type=int,
        default=50,
        required=True,
        help="Number of labeled examples to use for training.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        required=True,
        help="The sampling method to use. Options: 'random', 'consensus_ds', 'consensus_dt', 'baseline_all', 'baseline_none'",
    )
    parser.add_argument(
        "--n_unlabeled",
        type=int,
        default=0,
        help="Number of unlabeled examples to select",
    )
    parser.add_argument(
        "--bracket",
        type=str,
        default="",
        help="Sampling bracket. Options: [0, 40], [20, 60], [40, 80], [60, 100]",
    )
    args = parser.parse_args()

    validate_arguments(args)

    main(
        model=args.model,
        dataset_to_process=args.dataset,
        seed=args.seed,
        n_labeled=args.n_labeled,
        sampling_method=args.sampling_method,
        n_unlabeled=args.n_unlabeled,
        bracket=args.bracket,
    )
