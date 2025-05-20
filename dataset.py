import json
import os

import numpy as np
import openml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_cc18_dataset(dataset_id):
    """
    Loads a dataset from the OpenML repository by dataset ID.

    Parameters:
        dataset_id (int): OpenML dataset ID.

    Returns:
        tuple: (x, y)
            x (DataFrame): Feature set.
            y (array-like): Target labels as a numpy array.
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    y = y.to_numpy()
    return x, y


def label_encode(y):
    """
    Encodes categorical labels using a LabelEncoder.
    Done for two reasons:
        1. Converts categorical data into numerical data
        2. If the labels are numerical, maps them to an array starting at 0 filled with consecutive numbers.

    Parameters:
        y (array-like): Array of labels to encode.

    Returns:
        array-like: Encoded numeric labels.
    """
    le = LabelEncoder()
    return le.fit_transform(y)


def preprocess_categorical_features(x):
    """
    Encodes categorical features in a DataFrame into numeric format.
    At the moment, the LabelEncoder is chosen as a way to do this.
    Other, more complex ways can be introduced here

    Parameters:
        x (DataFrame): Feature set with categorical columns.

    Returns:
        array: Feature set with categorical columns encoded as integers.
    """
    for col in x.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))  # Ensure strings for consistency
    x = x.to_numpy()
    return x


def load_and_store_data(dataset_id):
    (
        x,
        y,
        categorical_columns,
        categorical_col_indices,
        numerical_columns,
        numerical_col_indices,
        unique_vals_within_category,
    ) = load_data(dataset_id)
    np.save(f"datasets/{dataset_id}_x.npy", x)
    np.save(f"datasets/{dataset_id}_y.npy", y)
    with open(f"datasets/{dataset_id}_categorical_columns.json", "w") as f:
        json.dump(categorical_columns, f)
    np.save(
        f"datasets/{dataset_id}_categorical_col_indices.npy",
        np.array(categorical_col_indices),
    )
    with open(f"datasets/{dataset_id}_numerical_columns.json", "w") as f:
        json.dump(numerical_columns, f)
    np.save(
        f"datasets/{dataset_id}_numerical_col_indices.npy",
        np.array(numerical_col_indices),
    )
    np.save(
        f"datasets/{dataset_id}_unique_vals_within_category.npy",
        np.array(unique_vals_within_category),
    )


def load_data(dataset_id, preprocess_categorical=True):
    """
    Loads a dataset and optionally preprocesses categorical features.

    Parameters:
        dataset_id (int): OpenML dataset ID.
        preprocess_categorical (bool): Whether to preprocess categorical features.

    Returns:
        tuple: (x, y)
            x (array): Preprocessed feature set.
            y (array): Encoded target labels.
    """
    # Read the downloaded dataset if it exists, otherwise download from OpenML
    if os.path.exists(f"datasets/{dataset_id}_x.npy"):
        x = np.load(f"datasets/{dataset_id}_x.npy", allow_pickle=True)
        y = np.load(f"datasets/{dataset_id}_y.npy", allow_pickle=True)
        with open(f"datasets/{dataset_id}_categorical_columns.json", "r") as f:
            categorical_columns = json.load(f)
        categorical_col_indices = np.load(
            f"datasets/{dataset_id}_categorical_col_indices.npy", allow_pickle=True
        ).tolist()
        with open(f"datasets/{dataset_id}_numerical_columns.json", "r") as f:
            numerical_columns = json.load(f)
        numerical_col_indices = np.load(
            f"datasets/{dataset_id}_numerical_col_indices.npy", allow_pickle=True
        ).tolist()
        unique_vals_within_category = np.load(
            f"datasets/{dataset_id}_unique_vals_within_category.npy", allow_pickle=True
        ).tolist()

    else:
        print("Downloading dataset...")
        x, y = load_cc18_dataset(dataset_id)

        # identify categorical columns
        categorical_columns = x.select_dtypes(
            include=["object", "category"]
        ).columns.to_list()
        categorical_col_indices = [
            x.columns.get_loc(col) for col in categorical_columns
        ]
        numerical_columns = x.drop(columns=categorical_columns).columns.to_list()
        numerical_col_indices = [x.columns.get_loc(col) for col in numerical_columns]

        unique_vals_within_category = []

        for col in categorical_columns:
            unique_vals_within_category.append(len(x[col].unique()))

        y = label_encode(y)

        if preprocess_categorical:
            x = preprocess_categorical_features(x)

    return (
        x,
        y,
        categorical_columns,
        categorical_col_indices,
        numerical_columns,
        numerical_col_indices,
        unique_vals_within_category,
    )


def labeled_unlabeled_split(x_train, y_train, n_labeled, seed):
    """
    Splits the training dataset into labeled and unlabeled sets, maintaining class balance.

    Parameters:
        x_train (array-like): Training features.
        y_train (array-like): Training labels.
        n_labeled (int): Total number of labeled samples to retain.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (x_train_lab, x_train_unlab, y_train_lab, y_train_unlab)
            x_train_lab (array): Labeled training features.
            x_train_unlab (array): Unlabeled training features.
            y_train_lab (array): Labeled training labels.
            y_train_unlab (array): Unlabeled training labels.
    """
    np.random.seed(seed)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)

    labeled_per_class = {cls: n_labeled // n_classes for cls in unique_classes}
    remainder = n_labeled % n_classes

    for cls in unique_classes[:remainder]:
        labeled_per_class[cls] += 1

    labeled_indices = []
    unlabeled_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        np.random.shuffle(cls_indices)
        labeled_cls_indices = cls_indices[: labeled_per_class[cls]]
        unlabeled_cls_indices = cls_indices[labeled_per_class[cls] :]

        labeled_indices.extend(labeled_cls_indices)
        unlabeled_indices.extend(unlabeled_cls_indices)

    x_train_lab = x_train[labeled_indices]
    y_train_lab = y_train[labeled_indices]
    x_train_unlab = x_train[unlabeled_indices]
    y_train_unlab = np.full(len(unlabeled_indices), np.nan)

    return x_train_lab, x_train_unlab, y_train_lab, y_train_unlab


def create_performance_dict(
    y_test, y_pred, y_pred_proba, std_dev, sampling_time, model_time
):
    """
    Creates a dictionary of performance metrics based on model predictions.

    Parameters:
        y_test (array-like): True labels for the test set.
        y_pred (array-like): Predicted labels for the test set.
        y_pred_proba (array-like): Predicted probabilities for the test set.
        std_dev (float): Standard deviation for sampling score distribution.
        sampling_time (float): Time taken to sample unlabeled data points.
        model_time (float): Time taken to train the model.

    Returns:
        dict: Dictionary containing performance metrics.
            - accuracy (float): Classification accuracy.
            - auprc_macro (float): Macro-averaged AUPRC.
            - auprc_weighted (float): Weighted AUPRC.
            - auc_ovo (float): One-vs-One ROC AUC score.
            - f1_macro (float): Macro-averaged F1 score.
            - f1_weighted (float): Weighted F1 score.
            - precision_macro (float): Macro-averaged precision score.
            - precision_weighted (float): Weighted precision score.
            - recall_macro (float): Macro-averaged recall score.
            - recall_weighted (float): Weighted recall score.
            - sampling_std_dev (float): Standard deviation for sampling score distribution.
            - sampling_time (float): Time taken to sample unlabeled data points.
            - model_time (float): Time taken to train the model.
    """
    if y_pred_proba.shape[1] == 2:
        # in the case of binary classification:
        y_pred_proba = y_pred_proba[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auprc_macro = average_precision_score(y_test, y_pred_proba, average="macro")
    auprc_weighted = average_precision_score(y_test, y_pred_proba, average="weighted")
    auc_ovo = roc_auc_score(y_test, y_pred_proba, multi_class="ovo")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")

    perf_dict = {
        "accuracy": accuracy,
        "auprc_macro": auprc_macro,
        "auprc_weighted": auprc_weighted,
        "auc_ovo": auc_ovo,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "sampling_std_dev": std_dev,
        "sampling_time": sampling_time,
        "model_time": model_time,
        "total_time": model_time + sampling_time,
    }

    return perf_dict


def balanced_train_val_split(x_train, y_train, test_size=0.2, random_state=42):
    """
    Splits data into training and validation sets ensuring all classes appear in training data.

    Parameters:
        x_train (array-like): Training features
        y_train (array-like): Training labels
        test_size (float): Proportion of data for validation (default: 0.2)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (x_train_final, x_val, y_train_final, y_val)
            x_train_final (array): Training features with all classes
            x_val (array): Validation features
            y_train_final (array): Training labels with all classes
            y_val (array): Validation labels
    """

    np.random.seed(random_state)
    # Identify unique classes
    classes = np.unique(y_train)

    # Get indices for guaranteed samples (at least one per class)
    train_indices = []
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        train_indices.extend(cls_indices[:1])  # Take at least one sample per class

    # Get remaining indices
    remaining_indices = [i for i in range(len(y_train)) if i not in train_indices]
    remaining_x = x_train[remaining_indices]
    remaining_y = y_train[remaining_indices]

    # Determine test size for remaining data to achieve original test_size proportion
    if len(remaining_indices) > 0:
        adjusted_test_size = (len(y_train) * test_size - 0) / len(remaining_indices)
        adjusted_test_size = min(
            max(adjusted_test_size, 0), 1
        )  # Ensure it's between 0 and 1

        # Split remaining data
        try:
            # Try stratified sampling first
            x_remaining_train, x_val, y_remaining_train, y_val = train_test_split(
                remaining_x,
                remaining_y,
                stratify=remaining_y,
                test_size=0.2,
                random_state=random_state,
            )
        except ValueError as e:
            # Fall back to non-stratified sampling if stratified fails
            x_remaining_train, x_val, y_remaining_train, y_val = train_test_split(
                remaining_x, remaining_y, test_size=0.2, random_state=random_state
            )

        # Combine guaranteed samples with remaining train samples
        x_train_final = np.vstack([x_train[train_indices], x_remaining_train])
        y_train_final = np.concatenate([y_train[train_indices], y_remaining_train])
    else:
        # If no remaining samples, use all for training
        x_train_final = x_train[train_indices]
        y_train_final = y_train[train_indices]
        x_val = np.array([])
        y_val = np.array([])

    return x_train_final, x_val, y_train_final, y_val
