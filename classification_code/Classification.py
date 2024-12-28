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

from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedKFold
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

def balance_dataset(X, y, method='random_oversampling'):
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
                               )
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
                            )
            balanced_dfs.append(cls_df)
        
        df_balanced = pd.concat(balanced_dfs)
        return df_balanced.drop('target', axis=1), df_balanced['target']
    
    elif method == 'smote':
        # Use SMOTE for more sophisticated oversampling
        smote = SMOTE()
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
        optimizers_dict: dict = {str(torch.optim.AdamW): {'betas': (0.9, 0.999), 'eps': 1e-8}},
        hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['class_weights', 'act_fn'])
        
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

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.class_weights = class_weights
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.f1_score = torchmetrics.F1Score(task='binary', num_classes=2)
        self.auroc = torchmetrics.AUROC(task='binary', num_classes=2)

    def _init_weights(self, layer):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            self.parameters(),
            **self.hparams.config_optimizer,
            **self.hparams.optimizers_dict[str(self.hparams.optimizer)]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
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
        y = y.float().view(-1, 1)

        
        
        # Handle class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[y.long().squeeze()]  # Remove extra dimension for indexing
            loss = F.binary_cross_entropy(y_hat.squeeze(), y.squeeze(), weight=weights)
        else:
            loss = F.binary_cross_entropy(y_hat.squeeze(), y.squeeze())
            
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        # Avoid updating metrics if we have only one class
        if len(torch.unique(y)) > 1:
            self.train_acc(y_hat, y)
            self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        # Avoid updating metrics if we have only one class
        if len(torch.unique(y)) > 1:
            self.val_acc(y_hat, y)
            self.f1_score(y_hat, y)
            self.auroc(y_hat, y)
            self.log('val_acc', self.val_acc, on_epoch=True)
            self.log('val_f1', self.f1_score, on_epoch=True)
            self.log('val_auroc', self.auroc, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        # Log learning rate
        self.log('learning_rate', optimizer.param_groups[0]['lr'])

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
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)

    max_epochs = kwargs.get('max_epochs', 100)
    batch_size = kwargs.get('batch_size', 64)
    monitor = kwargs.get('monitor', 'val_loss')
    trainer_configs = kwargs.get('trainer_configs', {})

    del kwargs['trainer_configs']
    del kwargs['monitor']
    del kwargs['max_epochs']
    del kwargs['batch_size']


    fold_metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'val_auroc': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(latent_data, labels)):
        print(f"Fold {fold + 1}")
        train_data = latent_data[train_idx]
        train_labels = labels[train_idx]
        val_data = latent_data[val_idx]
        val_labels = labels[val_idx]
        class_counts = torch.bincount(train_labels.long())
        min_class_count = torch.min(class_counts).item()
        batch_size = min(batch_size, min_class_count)
        batch_size = max(batch_size, 2)

        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        

        _, label_counts = np.unique(train_labels, return_counts=True)
        weight_per_class = 1. / label_counts
        weights = np.array([weight_per_class[t] for t in train_labels])
        weights = torch.from_numpy(weights).float()

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_labels),
                replacement=True
            ),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True
        )
        
        # Calculate class weights
        class_weights = len(train_labels) / (2 * class_counts)
        # Initialize and train classifier
        classifier = NNClassifier(
            input_dim=train_data.shape[1],
            class_weights=class_weights,
            **kwargs
        )
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(patience=10, monitor=monitor),
                # ModelCheckpoint(
                #     monitor='val_loss',
                #     dirpath='models/classifier',
                #     filename='best-classifier'
                # )
            ],
            **trainer_configs,
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

        torch.cuda.synchronize()
        del train_loader, val_loader
        del classifier, trainer
    
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

        final_metrics.update({
            "val_loss": mean_metrics['val_loss'],
            "val_f1": mean_metrics['val_f1'],
            "val_auroc": mean_metrics['val_auroc']
        })

        train.report(
            checkpoint=checkpoint,
            metrics=final_metrics
        )


    
    return final_metrics


def train_classifier_wrapper(tune_configs: dict, norm_configs_id: ray.ObjectRef):
    norm_configs = ray.get(norm_configs_id)
    return train_classifier(**tune_configs, **norm_configs)

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
        "optimizers_dict": {
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
        "cpu": 4,
        "gpu": 0.25 
    }

    try:


        ray.init(
            num_cpus=cpu_count,
            num_gpus=1,
        )

        norm_configs_id = ray.put(norm_configs)

        analysis = tune.run(
            partial(train_classifier_wrapper, norm_configs_id=norm_configs_id),
            config=tune_configs,
            resources_per_trial=resources_per_trial,
            num_samples=num_samples,
            storage_path=storage_path,
            keep_checkpoints_num=1,
            checkpoint_score_attr='combined_metrics',
            stop = {
                "val_loss": 0.01,
            },
            verbose=2
            
        )

        # Enhanced results analysis
        best_trial = analysis.get_best_trial("combined_metrics", "min", "last")
        best_config = best_trial.config
        best_checkpoint = analysis.get_best_checkpoint(best_trial, "combined_metrics", "min")

        # Log detailed results
        print(f"Best trial config: {best_config}")
        print(f"Best combined metrics: {best_trial.last_result['combined_metrics']}")
        print(f"Best validation loss: {best_trial.last_result['val_loss']}")
        print(f"Best validation F1: {best_trial.last_result['val_f1']}")
        print(f"Best validation AUROC: {best_trial.last_result['val_auroc']}")
        print(f"Best checkpoint path: {best_checkpoint}")

        return best_trial, best_config, best_checkpoint
    except Exception:
        raise
    finally:
        ray.shutdown()




