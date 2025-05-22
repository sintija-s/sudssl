import time
import warnings
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset


class SamplingMethod:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def __str__(self):
        return self.name

    def score(self, x_train_lab, x_train_unlab, y_train_lab, seed):
        """
        Abstract method to compute scores for data points.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Score function must be implemented in subclass.")

    def sample(self, x_train_lab, x_train_unlab, y_train_lab, n_sample, seed,
               bracket=None, mean=None, sampling_strategy='highest'):
        """
        Sample data points based on the selected sampling strategy.
        
        Parameters:
            x_train_lab: labeled training data
            x_train_unlab: unlabeled training data
            y_train_lab: labels for labeled data
            n_sample: number of samples to select
            seed: random seed for reproducible results
            bracket: (lower, upper) limits to filter scores
            mean: target score mean for Gaussian or closest-to-value sampling
            sampling_strategy: 'highest' (default), 'bracket', 'percentile', 'gaussian', or 'closest-to-value'
    
        Returns:
            tuple: (x_sampled, y_sampled, std_dev)
                x_sampled (ndarray): sampled unlabeled data points
                y_sampled (array): corresponding labels; constant -1
                std_dev (float): standard deviation of sampling scores
        """
        np.random.seed(seed)
        elapsed_time = 0
        start_time = time.time()
        scores = self.score(x_train_lab, x_train_unlab, y_train_lab, seed)
        elapsed_time += time.time() - start_time
        
        if sampling_strategy == 'highest':
            # Top-n random selection (default)
            start_time = time.time()
            idcs = np.argsort(scores)[-n_sample:]
            elapsed_time += time.time() - start_time
        
        elif sampling_strategy == 'bracket':
            if not bracket:
                raise ValueError("Bracket must be specified for 'bracket' sampling.")
            lower, upper = bracket
            start_time = time.time()
            valid_indices = np.where((scores >= lower) & (scores <= upper))[0]
            if len(valid_indices) == 0:
                warnings.warn(f"Brackets are too tight; no scores fall inside the bracket: {scores}")
                return np.empty((0, x_train_lab.shape[1])), np.empty(0), 0
            elif len(valid_indices) < n_sample:
                warnings.warn(f"Only {len(valid_indices)} points found within the bracket; sampling only those.")
                idcs = np.random.choice(valid_indices, size=len(valid_indices), replace=True)
            else:
                idcs = np.random.choice(valid_indices, size=n_sample, replace=False)
            elapsed_time += time.time() - start_time
        
        elif sampling_strategy == 'percentile':
            if not bracket:
                raise ValueError("Bracket must be specified for 'percentile' sampling.")
            lower, upper = bracket
            start_time = time.time()
            lower_threshold = np.percentile(scores, lower)
            upper_threshold = np.percentile(scores, upper)
            valid_indices = np.where((scores >= lower_threshold) & (scores <= upper_threshold))[0]
            if len(valid_indices) < n_sample:
                warnings.warn(f"Only {len(valid_indices)} points found within the percentile; sampling only those.")
                idcs = np.random.choice(valid_indices, size=len(valid_indices), replace=True)
            else:
                idcs = np.random.choice(valid_indices, size=n_sample, replace=False)
            elapsed_time += time.time() - start_time
        
        elif sampling_strategy == 'gaussian':
            if mean is None:
                raise ValueError("Mean must be specified for 'gaussian' sampling.")
            # Automatically calculate std_dev such that ~2*n_sample points are within [-3std, 3std]
            start_time = time.time()
            valid_scores = scores[np.argsort(np.abs(scores - mean))[:2 * n_sample]]
            std_dev = (valid_scores.max() - valid_scores.min()) / 6
            std_dev = np.max((std_dev, 0.001))
            gaussian_distribution = norm(loc=0, scale=std_dev)
            
            probs = gaussian_distribution.pdf(np.abs(scores - mean))
            if np.max(probs) == 0:
                warnings.warn(f"No scores around that mean.")
                return np.empty((0, x_train_lab.shape[1])), np.empty(0), 0
            probs /= probs.sum()  # Normalize probabilities
            idcs = np.random.choice(len(scores), size=n_sample, p=probs)
            elapsed_time += time.time() - start_time
    
        elif sampling_strategy == 'closest-to-value':
            if mean is None:
                raise ValueError("Mean must be specified for 'closest-to-value' sampling.")
            start_time = time.time()
            idcs = np.argsort(np.abs(scores - mean))[:n_sample]
            elapsed_time += time.time() - start_time
        
        else:
            raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
        return x_train_unlab[idcs], [-1]*len(idcs), np.std(scores[idcs]), elapsed_time

# Baseline: Sample no unlabeled data points
class BaselineNone(SamplingMethod):
    def sample(self, x_train_lab, x_train_unlab, y_train_lab, n_sample, seed,
               bracket=None, mean=None, sampling_strategy='highest'):
        return np.empty((0, x_train_lab.shape[1])), np.empty(0), 0, 0 

# Baseline: Use all unlabeled data points
class BaselineAll(SamplingMethod):
    def sample(self, x_train_lab, x_train_unlab, y_train_lab, n_sample, seed,
               bracket=None, mean=None, sampling_strategy='highest'):
        return x_train_unlab, [-1]*len(x_train_unlab), 0, 0
        
# Select random unlabeled data points
class RandomSampling(SamplingMethod):
    def sample(self, x_train_lab, x_train_unlab, y_train_lab, n_sample, seed,
               bracket=None, mean=None, sampling_strategy='highest'):
        np.random.seed(seed)
        start_time = time.time()
        indices = np.random.choice(len(x_train_unlab), size=n_sample, replace=False)
        end_time = time.time()
        return x_train_unlab[indices], [-1]*len(indices), 0, end_time - start_time


class ConsensusSampling(SamplingMethod):
    def __init__(self, clf_base='DS', n_clf_base=10, disagreement_type='entropy', ds_max_depth=2):
        super().__init__()
        #base classifier must be decision tree (DT) or autoencoder (AE) or decision stump (DS)
        if clf_base not in ['DT', 'AE', 'DS']: 
            raise ValueError("Base classifier must be in ['DT', 'AE', 'DS']")
        if disagreement_type not in ['entropy', 'most_common_ratio']:
            raise ValueError("Disagreement type must be in ['entropy', 'most_common_ratio']")
        self.disagreement_type = disagreement_type
        self.clf_base_name = clf_base
        self.n_clf_base = n_clf_base # number of base classifiers
        self.clfs_base = [] # list of base classifiers
        self.clfs_feature_subsets = [] # list of feature subsets
        self.ds_max_depth = ds_max_depth # max depth for decision stump; will be ignored for other clf_base
        self.name += f"({clf_base})"

    # calculate disagreement score: 
    # 1 - count_most_frequent_prediction / count_all_predictions
    def most_common_ratio(predictions):
        # For each column, count how often the most common value occurs
        def most_common_ratio(col):
            counts = np.bincount(col)
            return np.max(counts) / len(col)
            
        return 1 - np.apply_along_axis(most_common_ratio, axis=0, arr=np.array(predictions))

    def vote_entropy(predictions):
        # predictions: shape (n_models, n_samples)
        n_models = np.array(predictions).shape[0]
    
        def normalized_entropy_for_column(col):
            values, counts = np.unique(col, return_counts=True)
            probs = counts / n_models
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(values)) if len(values) > 1 else 1  # avoid div by 0
            return entropy / max_entropy
    
        return np.abs(np.apply_along_axis(normalized_entropy_for_column, axis=0, arr=np.array(predictions)))
        
    def score(self, x_train_lab, x_train_unlab, y_train_lab, seed):
        rng = np.random.default_rng(seed)
        n_features = x_train_lab.shape[1]
        n_bootstrap = int(len(x_train_lab)/2)
        predictions = []
        x_train_concat = np.concatenate((x_train_lab, x_train_unlab))
        
        for n in range(self.n_clf_base):
            # Create bootstrap sample of the data
            bootstrap_indices = rng.choice(n_bootstrap, size=n_bootstrap, replace=True)
            x_bootstrap = x_train_lab[bootstrap_indices]
            y_bootstrap = y_train_lab[bootstrap_indices]

            # draw sqrt(n_features) without replacement
            feat_idx = rng.choice(n_features, size=int(sqrt(n_features)/2), replace=False)
            self.clfs_feature_subsets.append(feat_idx)
            
            # train base classifier & predict unlabeled set
            if self.clf_base_name == 'DT':
                clf = DecisionTreeClassifier(random_state=seed + n)
                clf.fit(x_bootstrap[:, feat_idx], y_bootstrap)
                # clf.fit(x_train_lab[:, feat_idx], y_train_lab)
                self.clfs_base.append(clf)
                predictions.append(clf.predict(x_train_unlab[:, feat_idx]))
            elif self.clf_base_name == 'DS':
                clf = DecisionTreeClassifier(random_state=seed + n, max_depth=self.ds_max_depth)
                clf.fit(x_bootstrap[:, feat_idx], y_bootstrap)
                # clf.fit(x_train_lab[:, feat_idx], y_train_lab)
                self.clfs_base.append(clf)
                predictions.append(clf.predict(x_train_unlab[:, feat_idx]))
            elif self.clf_base_name == 'AE':

                # estimate latent dimension using PCA
                start_pca = time.perf_counter()
                pca = PCA().fit(x_train_concat[:, feat_idx]) # draw sample?
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                latent_dim = np.argmax(cumulative_variance >= 0.95) + 1

                # Data preparation
                x_tensor = torch.tensor(x_train_concat[:, feat_idx], dtype=torch.float32)
                dataset = TensorDataset(x_tensor)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                # Autoencoder training
                ae = Autoencoder(len(feat_idx), latent_dim)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
        
                best_loss = float('inf')
                epochs_no_improve = 0
                patience = 5

                for epoch in range(100):
                    # Manually reshuffle the dataset each epoch
                    shuffled_indices = torch.randperm(len(dataset))
                    shuffled_dataset = TensorDataset(x_tensor[shuffled_indices])
                    loader = DataLoader(shuffled_dataset, batch_size=32)
                    
                    ae.train()
                    epoch_loss = 0
                    for batch in loader:
                        x_batch = batch[0]
                        optimizer.zero_grad()
                        reconstructed = ae(x_batch)
                        loss = criterion(reconstructed, x_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * x_batch.size(0)
        
                    epoch_loss /= len(loader.dataset)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model = ae.state_dict()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            break
                ae.load_state_dict(best_model)
                ae.eval()

                # Reconstruction errors on x_train_unlab
                x_unlab_tensor = torch.tensor(x_train_unlab[:, feat_idx], dtype=torch.float32)
                with torch.no_grad():
                    recon = ae(x_unlab_tensor)
                    errors = ((x_unlab_tensor - recon) ** 2).mean(dim=1).numpy()
                    predictions.append(errors)

        if self.clf_base_name == 'AE':
            scores = np.mean(predictions, axis=0)
        else:
            if self.disagreement_type == 'entropy':
                scores = ConsensusSampling.vote_entropy(predictions)
            elif self.disagreement_type == 'most_common_ratio':
                scores = ConsensusSampling.most_common_ratio(predictions)
        
        return scores


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

