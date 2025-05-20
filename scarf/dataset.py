import numpy as np
import torch
from torch.utils.data import Dataset


class SCARFDataset(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        # Get the maximum values for each feature
        max_vals = self.data.max(axis=0)
        min_vals = self.features_low
        
        # Add a small epsilon where max == min to ensure max > min
        epsilon = 1e-6
        equal_mask = max_vals == min_vals
        max_vals[equal_mask] = min_vals[equal_mask] + epsilon
        
        return max_vals

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class SupervisedSCARFDataset(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        # Return both data and target
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.long) # Use torch.long for class labels
        return x, y

    def __len__(self):
        return len(self.data)