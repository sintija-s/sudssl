import optuna
import torch
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from models import extract_tabnet_embeddings
from dataset import create_performance_dict
from sklearn.model_selection import train_test_split
import config


def suggest_pretrain_tabnet_params(trial):
    """Suggest hyperparameters for TabNet pretrainer"""
    return {
        "n_d": trial.suggest_categorical("n_d", [8, 16, 32, 64, 128]),
        "n_a": trial.suggest_categorical("n_a", [8, 16, 32, 64, 128]),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0, step=0.5),
        # "lambda_sparse": trial.suggest_loguniform("lambda_sparse", 1e-6, 1e-1),
        "lambda_sparse": trial.suggest_categorical("lambda_sparse", [0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]),
        # "momentum":  trial.suggest_float("momentum", 0.6, 0.98),
        "momentum": trial.suggest_categorical("momentum", [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]),

        "optimizer_params": {
            # "lr": trial.suggest_loguniform("lr", 0.005, 0.025)
            "lr": trial.suggest_categorical("lr", [0.005, 0.01, 0.02, 0.025])
        },
        "n_shared": trial.suggest_int("n_shared", 1, 3),
        "n_independent": trial.suggest_int("n_independent", 1, 3),
        "pretraining_ratio": trial.suggest_categorical("pretraining_ratio", [0.2, 0.4, 0.5, 0.7, 0.8]),
    }


def suggest_tabnet_params(trial):
    """Suggest hyperparameters for TabNet classifier"""
    return {
        "n_d": trial.suggest_categorical("n_d", [8, 16, 32, 64, 128]),
        "n_a": trial.suggest_categorical("n_a", [8, 16, 32, 64, 128]),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0, step=0.5),
        # "lambda_sparse": trial.suggest_loguniform("lambda_sparse", 1e-6, 1e-1),
        "lambda_sparse": trial.suggest_categorical("lambda_sparse", [0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]),
        # "momentum":  trial.suggest_float("momentum", 0.6, 0.98),
        "momentum": trial.suggest_categorical("momentum", [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]),
        "optimizer_params": {
            # "lr": trial.suggest_loguniform("lr", 0.005, 0.025)
            "lr": trial.suggest_categorical("lr", [0.005, 0.01, 0.02, 0.025])
        },
        "n_shared": trial.suggest_int("n_shared", 1, 3),
        "n_independent": trial.suggest_int("n_independent", 1, 3),
    }


def pretrainer_objective(trial, X, y, cat_idxs, cat_dims, clf_X, clf_y, n_splits=3, seed=42):
    """
    Objective function for optimizing pretrainer hyperparameters.
    KFold is done on all data (X, y) for pretrainer training, but the linear probe (logistic regression)
    is always done on a train/val split of the labeled data (clf_X, clf_y).
    """
    params = suggest_pretrain_tabnet_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        try:
            # Adjust batch size to avoid issues with single samples
            n_samples = len(X_train)
            batch_size = min(256, max(16, n_samples // 10))
            while n_samples % batch_size == 1 and batch_size > 1:
                batch_size += 1
            
            # Create and train pretrainer
            pretrainer = TabNetPretrainer(
                seed=seed,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                lambda_sparse=params['lambda_sparse'],
                momentum=params['momentum'],
                optimizer_params=params['optimizer_params'],
                n_shared=params['n_shared'],
                n_independent=params['n_independent'],
                verbose=0,
            )
            
            # Fit pretrainer on training data
            pretrainer.fit(
                X_train=X_train,
                eval_set=[X_val],
                max_epochs=config.tabnet_parameters['max_epochs_pretrain'],
                patience=config.tabnet_parameters['patience'],
                batch_size=batch_size,
                virtual_batch_size=min(128, batch_size // 2) if batch_size > 2 else 1,
                pretraining_ratio=params['pretraining_ratio'],
            )
            # Extract embeddings for all labeled data
            labeled_embeddings = extract_tabnet_embeddings(pretrainer, clf_X)
            # Split labeled data into train/val for linear probe
            emb_train, emb_val, y_train_lab, y_val_lab = train_test_split(
                labeled_embeddings, clf_y, test_size=1/3, random_state=seed, stratify=clf_y
            )
            clf = LogisticRegression(random_state=seed, max_iter=1000)
            clf.fit(emb_train, y_train_lab)
            
            y_pred = clf.predict(emb_val)
            y_pred_proba = clf.predict_proba(emb_val)
            perf = create_performance_dict(y_val_lab, y_pred, y_pred_proba, std_dev=0, sampling_time=0, model_time=0)
            print('perf:', perf)
            scores.append(perf["auprc_macro"])
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            scores.append(-1)
    return np.mean(scores)


def classifier_objective(trial, X, y, best_pretrainer, cat_idxs, cat_dims, n_splits=3, seed=42):
    """
    Objective function for optimizing classifier hyperparameters.
    Uses a pretrained model and evaluates classification performance.
    """
    params = suggest_tabnet_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        try:
            # Adjust batch size
            n_samples = len(X_train)
            batch_size = min(256, max(16, n_samples // 10))
            while n_samples % batch_size == 1 and batch_size > 1:
                batch_size -= 1
            
            # Create classifier
            model = TabNetClassifier(
                seed=seed,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                lambda_sparse=params['lambda_sparse'],
                momentum=params['momentum'],
                optimizer_params=params['optimizer_params'],
                n_shared=params['n_shared'],
                n_independent=params['n_independent'],
                verbose=0,
            )
            
            # Fit with pretrained model
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_name=["valid"],
                from_unsupervised=best_pretrainer,
                eval_metric=["accuracy"],
                max_epochs=config.tabnet_parameters['max_epochs_classifier'],
                patience=config.tabnet_parameters['patience'],
                batch_size=batch_size,
                virtual_batch_size=min(128, batch_size // 2) if batch_size > 2 else 1,
                drop_last=False,
            )
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            # # DEBUG PRINTS
            # print('--- DEBUG ---')
            # print('Fold:', fold_idx)
            # print('y_val unique:', np.unique(y_val))
            # print('y_pred unique:', np.unique(y_pred))
            # print('y_pred_proba shape:', y_pred_proba.shape)
            # print('y_val:', y_val[:10])
            # print('y_pred:', y_pred[:10])
            # print('y_pred_proba:', y_pred_proba[:2])  # print first 2 rows for brevity

            # Create performance dict and use macro AUPRC as score
            perf = create_performance_dict(y_val, y_pred, y_pred_proba, std_dev=0, sampling_time=0, model_time=0)
            print('perf:', perf)
            scores.append(perf["auprc_macro"])
            
        except Exception as e:
            print(f"Error in fold {fold_idx}: {e}")
            print('y_val:', y_val)
            scores.append(0.0)
    
    return np.mean(scores)


def optimize_pretrainer(X, y, cat_idxs, cat_dims, clf_X, clf_y, n_trials=50, n_splits=3, seed=42):
    """
    Optimize pretrainer hyperparameters using Optuna.
    
    Returns:
        tuple: (best_params, best_score, best_pretrainer)
    """
    print("Starting pretrainer hyperparameter optimization...")
    
    study = optuna.create_study(direction="maximize", study_name="tabnet_pretrainer")
    study.optimize(
        lambda trial: pretrainer_objective(trial, X, y, cat_idxs, cat_dims, clf_X, clf_y, n_splits, seed),
        n_trials=n_trials
    )
    
    print(f"Best pretrainer params: {study.best_params}")
    print(f"Best pretrainer score: {study.best_value:.4f}")
    
    # Train final pretrainer with best params
    best_params = study.best_params
    best_pretrainer = TabNetPretrainer(
        seed=seed,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        lambda_sparse=best_params['lambda_sparse'],
        momentum=best_params['momentum'],
        optimizer_params={"lr": best_params['lr']},
        n_shared=best_params['n_shared'],
        n_independent=best_params['n_independent'],
        verbose=1,
    )
    
    # Train on full dataset
    n_samples = len(X)
    batch_size = min(256, max(16, n_samples // 10))
    while n_samples % batch_size == 1 and batch_size > 1:
        batch_size -= 1
    
    best_pretrainer.fit(
        X_train=X,
        max_epochs=config.tabnet_parameters['max_epochs_pretrain'],
        patience=config.tabnet_parameters['patience'],
        batch_size=batch_size,
        virtual_batch_size=min(128, batch_size // 2) if batch_size > 2 else 1,
        pretraining_ratio=best_params['pretraining_ratio'],
    )
    
    # Save the pretrained model
    pretrained_model_path = "best_pretrainer.zip"
    # best_pretrainer.save_model(pretrained_model_path)
    
    return study.best_params, study.best_value, pretrained_model_path, best_pretrainer


def optimize_classifier(X, y, pretrained_model_path, cat_idxs, cat_dims, best_pretrainer, n_trials=50, n_splits=3, seed=42):
    """
    Optimize classifier hyperparameters using a pretrained model.
    
    Returns:
        tuple: (best_params, best_score)
    """
    print("Starting classifier hyperparameter optimization...")
    
    study = optuna.create_study(direction="maximize", study_name="tabnet_classifier")
    study.optimize(
        lambda trial: classifier_objective(trial, X, y, best_pretrainer, cat_idxs, cat_dims, n_splits, seed),
        n_trials=n_trials
    )
    
    print(f"Best classifier params: {study.best_params}")
    print(f"Best classifier score: {study.best_value:.4f}")
    
    return study.best_params, study.best_value


def run_complete_hpo(
    pretrain_X, pretrain_y, clf_X, clf_y, cat_idxs, cat_dims, 
    pretrainer_trials=50, classifier_trials=100, n_splits=3, seed=42
):
    """
    Run complete hyperparameter optimization for both pretrainer and classifier.
    
    Args:
        pretrain_X: Feature matrix for pretrainer
        pretrain_y: Target labels for pretrainer
        clf_X: Feature matrix for classifier
        clf_y: Target labels for classifier
        cat_idxs: Indices of categorical features
        cat_dims: Cardinality of each categorical feature
        pretrainer_trials: Number of trials for pretrainer optimization
        classifier_trials: Number of trials for classifier optimization
        n_splits: Number of CV folds
        seed: Random seed
    
    Returns:
        dict: Results containing best parameters for both components
    """
    results = {}
    
    # Step 1: Optimize pretrainer
    print("="*50)
    print("STEP 1: Optimizing TabNet Pretrainer")
    print("="*50)
    
    pretrainer_params, pretrainer_score, pretrained_model_path, best_pretrainer = optimize_pretrainer(
        pretrain_X, pretrain_y, cat_idxs, cat_dims, clf_X, clf_y, pretrainer_trials, n_splits, seed
    )
    
    results['pretrainer'] = {
        'best_params': pretrainer_params,
        'best_score': pretrainer_score,
        'model_path': pretrained_model_path
    }
    
    # Step 2: Optimize classifier using best pretrainer
    print("\n" + "="*50)
    print("STEP 2: Optimizing TabNet Classifier")
    print("="*50)
    
    classifier_params, classifier_score = optimize_classifier(
        clf_X, clf_y, pretrained_model_path, cat_idxs, cat_dims, best_pretrainer, classifier_trials, n_splits, seed
    )
    
    results['classifier'] = {
        'best_params': classifier_params,
        'best_score': classifier_score
    }
    
    # Save results
    with open("tabnet_hpo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Pretrainer best score: {pretrainer_score:.4f}")
    print(f"Classifier best score: {classifier_score:.4f}")
    print("Results saved to: tabnet_hpo_results.json")
    
    return results


if __name__ == "__main__":
    # Example usage - replace with your actual data loading
    print("TabNet Hyperparameter Optimization Script")
    print("Please modify the main section to load your specific dataset")
    
    # Example data loading code (uncomment and modify as needed):
    # from dataset import load_data
    # dataset_id = 40536  # Replace with your dataset ID
    # X, y, _, cat_idxs, _, _, cat_dims = load_data(dataset_id)
    # 
    # # Run complete HPO
    # results = run_complete_hpo(
    #     X=X, 
    #     y=y, 
    #     cat_idxs=cat_idxs, 
    #     cat_dims=cat_dims,
    #     pretrainer_trials=30,
    #     classifier_trials=30,
    #     n_splits=3,
    #     seed=42
    # )