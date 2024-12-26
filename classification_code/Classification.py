import json
import tempfile
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchmetrics
import torch.nn.functional as F
import os
import ray
import psutil

from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from ray.train import Checkpoint
from ray import train, tune




def analyze_class_imbalance(y, plot=True):
    """
    Analyze class distribution in a classification dataset.
    
    Parameters:
    y: array-like
        Target labels
    plot: bool
        Whether to create a visualization
    """
    # Count instances per class
    class_counts = Counter(y)
    total_samples = len(y)
    
    # Calculate percentages
    class_percentages = {cls: (count/total_samples)*100 
                        for cls, count in class_counts.items()}
    
    # Print statistics
    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples ({class_percentages[cls]:.2f}%)")
    
    # Calculate imbalance ratio
    majority_class = max(class_counts.values())
    minority_class = min(class_counts.values())
    imbalance_ratio = majority_class / minority_class
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    
    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(class_counts.keys()), 
                   y=list(class_counts.values()))
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()
    
    return class_counts, imbalance_ratio

def balance_dataset(X, y, method='random_oversampling', random_state=42):
    """
    Balance a dataset using various methods.
    
    Parameters:
    X: array-like
        Feature matrix
    y: array-like
        Target labels
    method: str
        'random_oversampling', 'random_undersampling', or 'smote'
    
    Returns:
    X_balanced, y_balanced: Balanced dataset
    """
    method = method.lower()

    if method == 'random_oversampling':
        # Separate majority and minority classes
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Get the class counts
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        n_samples = class_counts[majority_class]
        
        # Oversample minority classes
        balanced_dfs = []
        for cls in class_counts.keys():
            cls_df = df[df['target'] == cls]
            if cls != majority_class:
                cls_df = resample(cls_df,
                                replace=True,
                                n_samples=n_samples,
                                random_state=random_state)
            balanced_dfs.append(cls_df)
        
        # Combine balanced datasets
        df_balanced = pd.concat(balanced_dfs)
        return df_balanced.drop('target', axis=1), df_balanced['target']
    
    elif method == 'random_undersampling':
        # Similar to oversampling but downsample majority class
        df = pd.DataFrame(X)
        df['target'] = y
        
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        n_samples = class_counts[minority_class]
        
        balanced_dfs = []
        for cls in class_counts.keys():
            cls_df = df[df['target'] == cls]
            if cls != minority_class:
                cls_df = resample(cls_df,
                                replace=False,
                                n_samples=n_samples,
                                random_state=random_state)
            balanced_dfs.append(cls_df)
        
        df_balanced = pd.concat(balanced_dfs)
        return df_balanced.drop('target', axis=1), df_balanced['target']
    
    elif method == 'smote':
        # Use SMOTE for more sophisticated oversampling
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    else:
        raise ValueError("Method must be 'random_oversampling', 'random_undersampling', or 'smote'")


class NNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        act_fn: nn.Module = nn.ReLU(),
        config_optimizer: dict = {'lr': 1e-3, 'weight_decay': 1e-5}, 
        hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                act_fn,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.class_weights = class_weights
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            self.parameters(),
            **self.hparams.config_optimizer
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Handle class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[y.long()]
            loss = F.binary_cross_entropy(y_hat, y.float().view(-1, 1), weight=weights)
        else:
            loss = F.binary_cross_entropy(y_hat, y.float().view(-1, 1))
            
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        
        # Log metrics
        self.train_acc(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        
        # Log metrics
        self.val_acc(y_hat, y)
        self.f1_score(y_hat, y)
        self.auroc(y_hat, y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
        self.log('val_f1', self.f1_score, on_epoch=True)
        self.log('val_auroc', self.auroc, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        return self(x)

def train_classifier(
    latent_data: torch.Tensor,
    labels: torch.Tensor,
    num_folds = 5,
    **kwargs
):
    """
    Train the classifier using latent data from autoencoder
    """
    torch.set_float32_matmul_precision('medium')

    kf = KFold(n_splits=num_folds, shuffle=True)

    fold_metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'val_auroc': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(latent_data)):
        print(f"Fold {fold + 1}")
        train_data = latent_data[train_idx]
        train_labels = labels[train_idx]
        val_data = latent_data[val_idx]
        val_labels = labels[val_idx]
        
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=kwargs.get('batch_size', 64),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=kwargs.get('batch_size', 64),
            shuffle=False,
            num_workers=4
        )
        
        # Calculate class weights
        class_counts = torch.bincount(train_labels.long())
        class_weights = len(train_labels) / (2 * class_counts)
        # Initialize and train classifier
        classifier = NNClassifier(
            input_dim=train_data.shape[1],
            class_weights=class_weights,
            **kwargs
        )
        
        trainer = pl.Trainer(
            max_epochs=kwargs.get('max_epochs', 100),
            accelerator='gpu',
            devices=1,
            callbacks=[
                EarlyStopping(patience=10, monitor=kwargs.monitor),
                # ModelCheckpoint(
                #     monitor='val_loss',
                #     dirpath='models/classifier',
                #     filename='best-classifier'
                # )
            ],
            **kwargs.get('trainer_configs', {}),
            enable_progress_bar=True
        )
        
        trainer.fit(
            classifier, 
            train_loader, 
            val_loader
        )

        fold_metrics['train_loss'].append(trainer.callback_metrics['train_loss'].item())
        fold_metrics['val_loss'].append( trainer.callback_metrics['val_loss'].item())
        fold_metrics['train_acc'].append(trainer.callback_metrics['train_acc'].item())
        fold_metrics['val_acc'].append(trainer.callback_metrics['val_acc'].item())
        fold_metrics['val_f1'].append(trainer.callback_metrics['val_f1'].item())
        fold_metrics['val_auroc'].append(trainer.callback_metrics['val_auroc'].item())
    
    mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}

    final_metrics = {
        "combined_metrics": mean_metrics['val_f1'] + mean_metrics['val_auroc'] + 0.5 * mean_metrics['val_loss'],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, 'metrics_classifier.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'mean_metrics': mean_metrics,
                'final_metrics': final_metrics
            }, f)
        
        checkpoint = Checkpoint.from_directory(tmpdir)

        train.report(
            checkpoint=checkpoint,
            metrics=final_metrics
        )


    
    return final_metrics


def train_classifier_wrapper(tune_config: dict, norm_config: dict):
    return train_classifier(**tune_config, **norm_config)

def main_classifier(latent_data: torch.Tensor, labels: torch.Tensor, num_samples=10, max_num_epochs=100):
    
    tune_configs = {
        "hidden_dims": tune.choice([[64, 32], [16, 8], [32], [64]]),
        "batch_size": tune.choice([32, 64, 128]),
        "act_fn": tune.choice([nn.ReLU(), nn.Tanh(), nn.GELU(), nn.LeakyReLU(), nn.Tanhshrink(), nn.SELU()]),
        "optimizer": tune.choice([torch.optim.AdamW, torch.optim.SGD]),
        "dropout": tune.uniform(0.05, 0.3),
        "config_optimizer": {
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "lr": tune.loguniform(1e-5, 1e-2),
        }
    }
    
    norm_configs = {
        "optimizer_configs": {
            str(torch.optim.AdamW): {
                "betas": (0.87, 0.99),
                "eps": 1e-9
            },
            str(torch.optim.SGD): {
                "momentum": 0.9,
                "nesterov": True
            },
        },
        "max_epochs": max_num_epochs,
        "num_folds": 5,
        "monitor": "val_loss",
        "latent_data": latent_data,
        "labels": labels,

        "trainer_configs": {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "enable_progress_bar": False,
            # "gradient_clip_val": 0.5,
        }
    }

    storage_path = os.path.abspath(os.path.join(os.getcwd(), 'models', 'ray_classifier'))
    os.makedirs(storage_path, exist_ok=True)
    cpu_count = psutil.cpu_count(logical=False)

    resources_per_trial = {
        "cpu": cpu_count // 4,
        "gpu": 1 / 4
    }

    ray.init(
        num_cpus=cpu_count,
        num_gpus=1,
    )

    # analysis = tune.run(
        
    


