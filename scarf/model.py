import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform


import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single-GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue
            x = layer(x)
        return x


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_low: int,
        features_high: int,
        dim_hidden_encoder: int,
        num_hidden_encoder: int,
        dim_hidden_head: int,
        num_hidden_head: int,
        corruption_rate: float = 0.6,
        dropout: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        set_seed(seed)

        self.encoder = MLP(input_dim, dim_hidden_encoder, num_hidden_encoder, dropout)
        self.pretraining_head = MLP(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout)

        # Ensure features_high > features_low for all dimensions
        features_low = torch.Tensor(features_low)
        features_high = torch.Tensor(features_high)
        
        # Add a small epsilon where max == min to ensure max > min
        epsilon = 1e-5
        equal_mask = features_high == features_low
        features_high[equal_mask] = features_low[equal_mask] + epsilon
        
        # Create uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(features_low, features_high)
        self.corruption_rate = corruption_rate

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true
        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # get embeddings
        embeddings = self.pretraining_head(self.encoder(x))
        embeddings_corrupted = self.pretraining_head(self.encoder(x_corrupted))

        return embeddings, embeddings_corrupted

    def get_embeddings_finetune(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)





import torch
import torch.nn as nn
from torch import Tensor

class SCARFClassifier(nn.Module):
    def __init__(
        self,
        scarf_model,
        num_classes: int,
        hidden_dim: int = None,
        num_hidden: int = 1,
        dropout: float = 0.0,
        freeze_encoder: bool = False,
        seed: int = 42,
    ) -> None:
        """SCARF Classifier for supervised fine-tuning.
        
        Args:
            scarf_model: Pre-trained SCARF model
            num_classes: Number of classes for classification
            hidden_dim: Dimension of hidden layers in the classification head. If None, uses the encoder's output dim.
            num_hidden: Number of hidden layers in the classification head
            dropout: Dropout probability
            freeze_encoder: Whether to freeze the pre-trained encoder
        """
        super().__init__()
        set_seed(seed)
        
        # Store the pre-trained SCARF model
        self.scarf = scarf_model
        self.freeze_encoder = freeze_encoder
        
        # Freeze the encoder if specified
        if freeze_encoder:
            for param in self.scarf.encoder.parameters():
                param.requires_grad = False
        
        # Get the dimension of the encoder's output
        encoder_dim = self.scarf.encoder.layers[-1].out_features
        
        # If hidden_dim is not provided, take half of the sum of the input and output dims
        if hidden_dim is None:
            hidden_dim = round((encoder_dim+num_classes)/2)
        
        # Create the classification head
        if num_hidden == 1:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, num_classes)
            )
        else:
            layers = []
            # First layer from encoder dimension to hidden dimension
            layers.append(nn.Linear(encoder_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            # Middle layers
            for _ in range(num_hidden - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, num_classes))
            
            self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        # Get embeddings from the encoder and clone them to allow for gradient computation
        if  self.freeze_encoder: # go into inference mode
            embeddings = self.scarf.get_embeddings(x)
        else:
            embeddings = self.scarf.get_embeddings_finetune(x)
        
        # Forward pass through the classification head
        logits = self.classifier(embeddings)
        
        return logits