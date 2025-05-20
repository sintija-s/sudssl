import sys
import time

import numpy as np
from util import set_seeds
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    classification_report,
)
from torch.optim import Adam
from torch.utils.data import DataLoader

import scarf
from scarf.dataset import SupervisedSCARFDataset

sys.path.append("../")
from scarf.loss import NTXent
from scarf.dataset import SCARFDataset
from scarf.utils import get_device, dataset_embeddings, pretrain_epoch


def extract_tabnet_embeddings(model, data):
    """Extract embeddings from a TabNet model (either classifier or pretrainer)"""
    # Convert to torch tensor if needed
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Set model to evaluation mode
    model.network.eval()

    # Extract embeddings
    with torch.no_grad():
        # Check model type and handle accordingly
        if isinstance(model, TabNetPretrainer):
            # TabNetPretrainer.forward_masks returns (embeddings, masks)
            embeddings, _ = model.network.forward_masks(data)
        else:
            # TabNetClassifier.forward_masks returns (output, embeddings, masks)
            _, embeddings, _ = model.network.forward_masks(data)

    embeddings_np = [emb.cpu().numpy() for emb in embeddings]

    return embeddings_np


def evaluate_pretraining_quality(
    unsupervised_model,
    x_train_lab,
    y_train_lab,
    x_train_unlab,
    x_val,
    y_val,
    x_test,
    y_test,
    n_labeled,
    n_sample,
    bracket,
    seed=42,
    output_dir="results/pretraining_evaluation",
    sampling_method="",
    model="tabnet",
):
    """
    Evaluate TabNet pretraining quality with t-SNE and linear probe
    """
    import os
    import csv

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(bracket, (list, tuple)) and len(bracket) == 2:
        bracket_min, bracket_max = bracket
    else:
        bracket_min, bracket_max = bracket, bracket  # If bracket is a single value

    dataset = output_dir.split("/")[1] if len(output_dir.split("/")) > 1 else "unknown"

    print("Extracting embeddings from pretrainer model...")
    train_unlab_emb = []

    if model == "tabnet":
        train_lab_emb = extract_tabnet_embeddings(unsupervised_model, x_train_lab)
        val_emb = extract_tabnet_embeddings(unsupervised_model, x_val)
        test_emb = extract_tabnet_embeddings(unsupervised_model, x_test)
        if x_train_unlab.shape[0] > 0:
            train_unlab_emb = extract_tabnet_embeddings(
                unsupervised_model, x_train_unlab
            )
    else:
        device = get_device()

        train_lab_ds = SCARFDataset(x_train_lab, y_train_lab)
        if x_train_unlab.shape[0] > 0:
            y_train_unlab = [-1] * len(x_train_unlab)
            train_unlab_ds = SCARFDataset(x_train_unlab, y_train_unlab)
        test_ds = SCARFDataset(x_test, y_test)
        val_ds = SCARFDataset(x_val, y_val)

        train_lab_loader = DataLoader(train_lab_ds, batch_size=128, shuffle=False)
        if x_train_unlab.shape[0] > 0:
            train_unlab_loader = DataLoader(
                train_unlab_ds, batch_size=128, shuffle=False
            )
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        train_lab_emb = dataset_embeddings(unsupervised_model, train_lab_loader, device)
        if x_train_unlab.shape[0] > 0:
            train_unlab_emb = dataset_embeddings(
                unsupervised_model, train_unlab_loader, device
            )
        val_emb = dataset_embeddings(unsupervised_model, val_loader, device)
        test_emb = dataset_embeddings(unsupervised_model, test_loader, device)

    # Prepare results list for CSV
    results = []

    # ----- CLUSTERING QUALITY EVALUATION -----
    print("Evaluating clustering quality...")

    # Combine labeled and unlabeled embeddings for clustering evaluation
    if x_train_unlab.shape[0] > 0:
        combined_emb = np.vstack([train_lab_emb, train_unlab_emb])
        combined_labels = np.concatenate([y_train_lab, [-1] * len(train_unlab_emb)])
    else:
        combined_emb = train_lab_emb.copy()
        combined_labels = y_train_lab.copy()

    valid_labels_mask = combined_labels != -1

    # Only calculate if we have at least 2 classes and enough samples
    try:
        # Clustering metrics require ground truth labels
        if len(np.unique(y_train_lab)) >= 2 and np.sum(valid_labels_mask) >= 2:
            silhouette = silhouette_score(
                combined_emb[valid_labels_mask], combined_labels[valid_labels_mask]
            )
            calinski = calinski_harabasz_score(
                combined_emb[valid_labels_mask], combined_labels[valid_labels_mask]
            )
            davies = davies_bouldin_score(
                combined_emb[valid_labels_mask], combined_labels[valid_labels_mask]
            )

            print(f"Clustering metrics (on labeled data):")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Index: {calinski:.4f}")
            print(f"Davies-Bouldin Index: {davies:.4f}")

            # Add clustering metrics to results
            results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "n_labeled": n_labeled,
                    "n_sample": n_sample,
                    "actually_sampled": len(x_train_unlab),
                    "sampling_method": sampling_method,
                    "bracket_min": bracket_min,
                    "bracket_max": bracket_max,
                    "trained_on": "embedding",
                    "metric": "silhouette_score",
                    "value": silhouette,
                }
            )
            results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "n_labeled": n_labeled,
                    "n_sample": n_sample,
                    "actually_sampled": len(x_train_unlab),
                    "sampling_method": sampling_method,
                    "bracket_min": bracket_min,
                    "bracket_max": bracket_max,
                    "trained_on": "embedding",
                    "metric": "calinski_harabasz_score",
                    "value": calinski,
                }
            )
            results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "n_labeled": n_labeled,
                    "n_sample": n_sample,
                    "actually_sampled": len(x_train_unlab),
                    "sampling_method": sampling_method,
                    "bracket_min": bracket_min,
                    "bracket_max": bracket_max,
                    "trained_on": "embedding",
                    "metric": "davies_bouldin_score",
                    "value": davies,
                }
            )
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")

    # ----- LINEAR PROBE EVALUATION -----
    print("Evaluating linear probe performance...")

    # Train a logistic regression classifier on training embeddings
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    clf.fit(train_lab_emb, y_train_lab)

    # Evaluate on validation data
    val_pred = clf.predict(val_emb)
    val_proba = clf.predict_proba(val_emb)

    val_acc = accuracy_score(y_val, val_pred)
    val_balanced_acc = balanced_accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    val_auprc = np.nan
    # if not all classes are present in the validation set, it will throw an error when trying to calculate probability based performance metrics
    try:
        if val_proba.shape[1] == 2:
            val_proba = val_proba[:, 1]
        val_auprc = average_precision_score(y_val, val_proba, average="macro")
    except Exception as e:
        print(f"Warning: Issue with AUPRC calculation - {e}")

    # Evaluate on test data
    test_pred = clf.predict(test_emb)
    test_proba = clf.predict_proba(test_emb)
    test_acc = accuracy_score(y_test, test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    if test_proba.shape[1] == 2:
        test_proba = test_proba[:, 1]
    test_auprc = average_precision_score(y_test, test_proba, average="macro")

    # Print results
    print("\n----- Linear Probe Performance -----")
    print(
        f"Validation: Accuracy={val_acc:.4f}, Balanced Accuracy={val_balanced_acc:.4f}, F1={val_f1:.4f}"
    )
    print(
        f"Test: Accuracy={test_acc:.4f}, Balanced Accuracy={test_balanced_acc:.4f}, F1={test_f1:.4f}"
    )
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, test_pred))

    # Add embedding results to CSV data
    for metric, value, split in [
        ("accuracy", val_acc, "validation"),
        ("balanced_accuracy", val_balanced_acc, "validation"),
        ("f1_score", val_f1, "validation"),
        ("auprc_macro", val_auprc, "validation"),
        ("accuracy", test_acc, "test"),
        ("balanced_accuracy", test_balanced_acc, "test"),
        ("f1_score", test_f1, "test"),
        ("auprc_macro", test_auprc, "test"),
    ]:
        results.append(
            {
                "dataset": dataset,
                "seed": seed,
                "n_labeled": n_labeled,
                "n_sample": n_sample,
                "actually_sampled": len(x_train_unlab),
                "sampling_method": sampling_method,
                "bracket_min": bracket_min,
                "bracket_max": bracket_max,
                "trained_on": "embedding",
                "metric": f"{metric}_{split}",
                "value": value,
            }
        )

    # ----- BASELINE COMPARISON -----
    # Train directly on raw features for comparison
    print("\nTraining baseline classifier on raw features...")

    baseline_clf = LogisticRegression(
        max_iter=1000, random_state=seed, class_weight="balanced"
    )
    baseline_clf.fit(x_train_lab, y_train_lab)

    # Evaluate on validation data
    baseline_val_pred = baseline_clf.predict(x_val)
    baseline_val_proba = baseline_clf.predict_proba(x_val)
    baseline_val_acc = accuracy_score(y_val, baseline_val_pred)
    baseline_val_balanced_acc = balanced_accuracy_score(y_val, baseline_val_pred)
    baseline_val_f1 = f1_score(y_val, baseline_val_pred, average="macro")
    baseline_val_auprc = np.nan
    try:
        if baseline_val_proba.shape[1] == 2:
            baseline_val_proba = baseline_val_proba[:, 1]
        baseline_val_auprc = average_precision_score(
            y_val, baseline_val_proba, average="macro"
        )
    except Exception as e:
        print(f"Warning: Issue with AUPRC calculation - {e}")

    baseline_test_pred = baseline_clf.predict(x_test)
    baseline_test_proba = baseline_clf.predict_proba(x_test)
    baseline_test_acc = accuracy_score(y_test, baseline_test_pred)
    baseline_test_balanced_acc = balanced_accuracy_score(y_test, baseline_test_pred)
    baseline_test_f1 = f1_score(y_test, baseline_test_pred, average="macro")
    if baseline_test_proba.shape[1] == 2:
        baseline_test_proba = baseline_test_proba[:, 1]
    baseline_test_auprc = average_precision_score(
        y_test, baseline_test_proba, average="macro"
    )

    print("\n----- Raw Features Baseline Performance -----")
    print(
        f"Test: Accuracy={baseline_test_acc:.4f}, Balanced Accuracy={baseline_test_balanced_acc:.4f}, F1={baseline_test_f1:.4f}"
    )

    # Add raw features results to CSV data
    for metric, value, split in [
        ("accuracy", baseline_val_acc, "validation"),
        ("balanced_accuracy", baseline_val_balanced_acc, "validation"),
        ("f1_score", baseline_val_f1, "validation"),
        ("auprc_macro", baseline_val_auprc, "validation"),
        ("accuracy", baseline_test_acc, "test"),
        ("balanced_accuracy", baseline_test_balanced_acc, "test"),
        ("f1_score", baseline_test_f1, "test"),
        ("auprc_macro", baseline_test_auprc, "test"),
    ]:
        results.append(
            {
                "dataset": dataset,
                "seed": seed,
                "n_labeled": n_labeled,
                "n_sample": n_sample,
                "actually_sampled": len(x_train_unlab),
                "sampling_method": sampling_method,
                "bracket_min": bracket_min,
                "bracket_max": bracket_max,
                "trained_on": "raw_features",
                "metric": f"{metric}_{split}",
                "value": value,
            }
        )
    # ----- SAVE RESULTS TO CSV FILE -----
    csv_file = f"{output_dir}/linear_probe_results_{sampling_method}_{n_labeled}le_{bracket}bracket_{n_sample}unl_{seed}seed_{model}.csv"
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "seed",
            "n_labeled",
            "n_sample",
            "actually_sampled",
            "sampling_method",
            "bracket_min",
            "bracket_max",
            "trained_on",
            "metric",
            "value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_scarf(
    x_train_lab,
    x_train_unlab,
    y_train_lab,
    y_train_unlab,
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
    ds,
    sampling_method,
):

    set_seeds(seed=seed)

    pretrain_x = np.concatenate((x_train_lab, x_train_unlab), axis=0)
    pretrain_y = np.concatenate((y_train_lab, y_train_unlab), axis=0)
    train_ds = scarf.dataset.SCARFDataset(pretrain_x, pretrain_y)

    n_pretrain_samples = len(pretrain_x)

    batch_size = 128
    while n_pretrain_samples % batch_size == 1:
        batch_size += 1
    epochs = 100
    device = scarf.utils.get_device()

    pretrain_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    start_time = time.time()

    pretrained_scarf = scarf.SCARF(
        input_dim=train_ds.shape[1],
        features_low=train_ds.features_low,
        features_high=train_ds.features_high,
        dim_hidden_encoder=8,
        num_hidden_encoder=3,
        dim_hidden_head=24,
        num_hidden_head=2,
        corruption_rate=0.6,
        dropout=0.1,
        seed=seed,
    ).to(device)

    optimizer = Adam(pretrained_scarf.parameters(), lr=1e-3, weight_decay=1e-5)
    ntxent_loss = NTXent()
    loss_history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = pretrain_epoch(
            pretrained_scarf, ntxent_loss, pretrain_loader, optimizer, device
        )
        loss_history.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"epoch {epoch}/{epochs} - loss: {loss_history[-1]:.4f}", end="\r")

    evaluate_pretraining_quality(
        pretrained_scarf,
        x_train_lab,
        y_train_lab,
        x_train_unlab,
        x_val,
        y_val,
        x_test,
        y_test,
        n_labeled,
        n_sample,
        bracket=bracket,
        seed=seed,
        output_dir=f"results/{ds}/pretraining_evaluation",
        sampling_method=sampling_method,
        model="scarf",
    )

    from scarf.model import SCARFClassifier

    num_classes = len(np.unique(y_train_lab))
    classifier = SCARFClassifier(
        scarf_model=pretrained_scarf,
        num_classes=num_classes,
        hidden_dim=None,  # The dimension of the hidden dimensions; None sets it to avg of input and output
        num_hidden=3,  # Number of hidden layers
        dropout=0.1,
        freeze_encoder=False,
        seed=seed,
    )

    while len(x_train_lab) % batch_size == 1:
        batch_size += 1

    supervised_train_ds = SupervisedSCARFDataset(x_train_lab, y_train_lab)
    supervised_val_ds = SupervisedSCARFDataset(x_val, y_val)
    supervised_test_ds = SupervisedSCARFDataset(x_test, y_test)
    device = scarf.get_device()  # Your function to get the appropriate device
    classifier = classifier.to(device)
    # 3. Set up optimization
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-5)

    train_loader = torch.utils.data.DataLoader(
        supervised_train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        supervised_val_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        supervised_test_ds, batch_size=batch_size, shuffle=True
    )

    # 4. Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = classifier(x)

            # Compute loss
            loss = criterion(logits, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    end_time = time.time()
    execution_time = end_time - start_time

    # 5. Evaluation on test set
    classifier.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)

            all_predictions.append(predicted.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())

    y_test_predictions = np.concatenate(all_predictions)
    y_test_probabilities = np.concatenate(all_probabilities)

    return y_test_predictions, y_test_probabilities, execution_time


def run_selfsl_tabnet(
    x_train_lab,
    x_train_unlab,
    y_train_lab,
    y_train_unl,
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
    ds,
    sampling_method,
):
    set_seeds(seed=seed)

    clf = TabNetClassifier(
        seed=seed,
        cat_idxs=categorical_column_idx,
        cat_dims=unique_vals_within_category,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 50, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        n_steps=3,
        gamma=1.3,
        n_d=16,
        n_a=16,
        lambda_sparse=1e-4,
        momentum=0.3,
    )

    start_time = time.time()

    pretrain_arr = np.concatenate((x_train_lab, x_train_unlab), axis=0)

    unsupervised_model = TabNetPretrainer(
        seed=seed,
        cat_idxs=categorical_column_idx,
        cat_dims=unique_vals_within_category,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 50, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        n_steps=3,
        gamma=1.3,
        n_d=16,
        n_a=16,
        lambda_sparse=1e-4,
        momentum=0.3,
    )

    # Calculate batch_size for pretrainer
    n_pretrain_samples = len(pretrain_arr)
    bs_pretrain = 256
    while n_pretrain_samples % bs_pretrain == 1:
        bs_pretrain += 1

    unsupervised_model.fit(
        X_train=pretrain_arr,
        pretraining_ratio=0.8,
        batch_size=bs_pretrain,
        patience=10,
        drop_last=False,
    )
    evaluate_pretraining_quality(
        unsupervised_model,
        x_train_lab,
        y_train_lab,
        x_train_unlab,
        x_val,
        y_val,
        x_test,
        y_test,
        n_labeled,
        n_sample,
        bracket=bracket,
        seed=seed,
        output_dir=f"results/{ds}/pretraining_evaluation",
        sampling_method=sampling_method,
        model="tabnet",
    )

    # Calculate batch_size for classifier
    n_clf_samples = len(x_train_lab)
    bs_clf = 64

    while n_clf_samples % bs_clf == 1:
        bs_clf += 1

    clf.fit(
        x_train_lab,
        y_train_lab,
        eval_set=[(x_val, y_val)],
        eval_name=["valid"],
        from_unsupervised=unsupervised_model,
        eval_metric=["accuracy"],
        patience=50,
        batch_size=bs_clf,
        drop_last=False,
    )

    end_time = time.time()
    execution_time = end_time - start_time

    clf.network.eval()
    with torch.no_grad():
        y_test_predictions = clf.predict(x_test)
        y_test_probabilities = clf.predict_proba(x_test)

    return y_test_predictions, y_test_probabilities, execution_time
