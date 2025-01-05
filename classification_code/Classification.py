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
import time


from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold, StratifiedKFold
from ray.train import Checkpoint
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from scipy import stats
from pathlib import Path



# Get root directory 
path = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = path.parent.absolute()



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
        output_dim=1,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        act_fn: nn.Module = nn.ReLU(),
        config_optimizer: dict = {'lr': 1e-3, 'weight_decay': 1e-5},
        optimizers_dict: dict = {str(torch.optim.AdamW): {'betas': (0.9, 0.999), 'eps': 1e-8}},
        hidden_dims: list = [64, 32],
        dropout: float = 0.3,
        monitor: str = 'val_loss'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['act_fn'])
        self.loss_function = nn.BCEWithLogitsLoss()
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
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
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

    def reset_config(self, new_config):
        """Reset model with new configuration"""
        # Update hyperparameters
        self.hparams.update(new_config)
        
        # Reset optimizer
        if 'config_optimizer' in new_config:
            self.configure_optimizers()
            
        # Reset dropout if changed
        if 'dropout' in new_config:
            for layer in self.model:
                if isinstance(layer, nn.Dropout):
                    layer.p = new_config['dropout']
                    
        return True
    
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
                "monitor": self.hparams.monitor,
            }
        }
    
    def load_state_dict(self, state_dict, strict=True):
        # Remove class_weights from state_dict if present
        if "class_weights" in state_dict:
            state_dict = state_dict.copy()
            self.class_weights = state_dict.pop("class_weights")
        return super().load_state_dict(state_dict, strict=strict)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.float().view(-1, 1)

        loss = self.loss_function(y_hat, y)
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

def compute_class_weights(train_labels):
    if torch.is_tensor(train_labels):
        labels = train_labels.cpu().numpy()
    else:
        labels = np.array(train_labels)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Get unique classes and their counts
    classes, counts = np.unique(labels, return_counts=True)
    
    # Compute weights per class
    weight_per_class = 1. / counts
    
    # Create weight mapping dictionary
    weight_dict = dict(zip(classes, weight_per_class))
    
    # Map weights to original labels
    weights = np.array([weight_dict[label] for label in labels])
        
    return torch.from_numpy(weights).float()


def train_classifier(
    data: torch.Tensor,
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
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')

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

    checkpoint = train.get_checkpoint()
    start = 0
    classifier = None
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            classifier = NNClassifier(
                input_dim=train_data.shape[1],
                **kwargs
            )
            classifier.load_state_dict(checkpoint_dict["model_state"])
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "r") as f:
                state = json.load(f)
                start = state['training_fold']
    


    for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels), start=start):
        print(f"Fold {fold + 1}")
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        val_data = data[val_idx]
        val_labels = labels[val_idx]
        class_counts = torch.bincount(train_labels.long())
        min_class_count = torch.min(class_counts).item()
        batch_size = min(batch_size, min_class_count)
        batch_size = max(batch_size, 2)

        logger_tensor = TensorBoardLogger(
            save_dir=os.path.join(ROOT_DIR, 'models', "cls_logs"),
            name=f"fold_{fold+1}",
            version=timestamp
        )

        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)

        weights = compute_class_weights(train_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                num_samples=len(train_labels),
                weights=weights,
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
        
        if classifier is None:
            # Initialize and train classifier
            classifier = NNClassifier(
                input_dim=train_data.shape[1],
                **kwargs
            )
        
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger_tensor,
            callbacks=[
                EarlyStopping(patience=2, monitor=monitor),
                # ModelCheckpoint(
                #     monitor='val_loss',
                #     dirpath='models/classifier',
                #     filename='best-classifier'
                # )
            ],
            **trainer_configs,
        )
        
        trainer.fit(classifier, train_loader, val_loader)
        
        fold_metrics['train_loss'].append(trainer.callback_metrics['train_loss'].item())
        fold_metrics['val_loss'].append( trainer.callback_metrics['val_loss'].item())
        fold_metrics['train_acc'].append(trainer.callback_metrics['train_acc'].item())
        fold_metrics['val_acc'].append(trainer.callback_metrics['val_acc'].item())
        fold_metrics['val_f1'].append(trainer.callback_metrics['val_f1'].item())
        fold_metrics['val_auroc'].append(trainer.callback_metrics['val_auroc'].item())

        torch.cuda.synchronize()
        del trainer
        # metric = {
        #     'combined_metrics': 0.5 * fold_metrics['val_f1'][-1] + 0.5 * fold_metrics['val_auroc'][-1] - 0.5 * fold_metrics['val_loss'][-1]
        # }
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                {"training_fold": fold, "model_state": classifier.state_dict()},
                os.path.join(temp_checkpoint_dir, "checkpoint.pt"),
            )
            # train.report(metrics=metric, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))
        
        torch.cuda.empty_cache()
        if train_loader: 
            del train_loader
        if val_loader:
            del val_loader
        if classifier:
            del classifier
        classifier = None

    mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}

    final_metrics = {
        "combined_metrics": 0.5 * mean_metrics['val_f1'] + 0.5 * mean_metrics['val_auroc'] - 0.5 * mean_metrics['val_loss'],
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

def final_train_classifier(data: torch.Tensor, labels: torch.Tensor, **kwargs):
    input_dim = kwargs.get('input_dim', 8)
    hidden_dims = kwargs.get('hidden_dims', [64, 32])
    batch_size = kwargs.get('batch_size', 64)
    act_fn = kwargs.get('act_fn', nn.ReLU())
    optimizer = kwargs.get('optimizer', torch.optim.AdamW)
    dropout = kwargs.get('dropout', 0.3)
    monitor = kwargs.get('monitor', 'train_loss')
    max_epochs = kwargs.get('max_epochs', 10)
    trainer_configs = kwargs.get('trainer_configs', {})
    optimizer_params = kwargs.get('optimizer_params', {'lr': 1e-3, 'weight_decay': 1e-5})
    model_save_path = kwargs.get('model_save_path', None)
    model_save_path = os.path.join(os.getcwd(), model_save_path) if model_save_path else None
    
    torch.set_float32_matmul_precision('medium')

    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    model_name = f"NNClassifier_{timestamp}"

    logger_tensor = TensorBoardLogger(
            save_dir=os.path.join(ROOT_DIR, 'models', "cls_logs"),
            name=model_name,
            version=timestamp
        )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor=monitor,
            patience=5,
            mode='min'
        ),
    ]

    train_dataset = TensorDataset(data, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    classifier = NNClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        optimizer=optimizer,
        act_fn=act_fn,
        config_optimizer=optimizer_params,
        dropout=dropout,
        monitor=monitor
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger_tensor,
        **trainer_configs
    )

    trainer.fit(classifier, train_loader)


    if model_save_path:
        print(f"Creating the folders for the model: {model_save_path}, if they do not exist.")
        print(f"With the model name: {model_name}")
        os.makedirs(model_save_path, exist_ok=True)
        model_path = os.path.join(model_save_path, model_name)
        print(f"Saving the model to: {model_path}")
        torch.save(classifier.state_dict(), model_path)

    torch.cuda.empty_cache()
    del train_loader, trainer

    return classifier

       

    


def train_classifier_wrapper(tune_configs: dict, norm_configs_id: ray.ObjectRef):
    norm_configs = ray.get(norm_configs_id)
    return train_classifier(**tune_configs, **norm_configs)

def main_classifier(data: torch.Tensor, labels: torch.Tensor, num_samples=10, max_num_epochs=100):
    
    tune_configs = {
        "hidden_dims": tune.choice([[64, 32], [16, 8], [32, 96, 16], [32], [64]]),
        "batch_size": tune.choice([32, 64, 128]),
        "act_fn": tune.choice([nn.Tanh(), nn.LeakyReLU(), nn.Tanhshrink(), nn.SELU()]),
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
        "data": data,
        "labels": labels,

        "trainer_configs": {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "enable_progress_bar": False,
            # "gradient_clip_val": 0.5,
        }
    }

    ray_init_configs = {
        'address': "local",
        "num_cpus": psutil.cpu_count(),
        "num_gpus": torch.cuda.device_count(),
        "ignore_reinit_error": True,
        "include_dashboard": False,  # Disable dashboard to avoid related errors,
    }

    storage_path = os.path.abspath(os.path.join(os.getcwd(), 'models', 'ray_classifier'))
    os.makedirs(storage_path, exist_ok=True)

    resources_per_trial = {
        "cpu": 4,
        "gpu": 0.25 
    }

    # asha_scheduler = tune.schedulers.ASHAScheduler(
    #     max_t=max_num_epochs,
    #     grace_period=2,
    #     reduction_factor=2,
    #     metric='combined_metrics',
    #     mode='max'
    # )

    pop_scheduler = PopulationBasedTraining(
        perturbation_interval=2,
        metric="combined_metrics",
        mode="max",
        hyperparam_mutations={
            "config_optimizer": {
                "lr": tune.loguniform(1e-6, 1e-1),
            },
            "dropout": tune.uniform(0.05, 0.5),
        },
        resample_probability=0.25,
        quantile_fraction=0.25
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if ray.is_initialized():
                ray.shutdown()

            context = ray.init(
                **ray_init_configs
            )
            print(context.dashboard_url)


            norm_configs_id = ray.put(norm_configs)

            analysis = tune.run(
                partial(train_classifier_wrapper, norm_configs_id=norm_configs_id),
                config=tune_configs,
                resources_per_trial=resources_per_trial,
                num_samples=num_samples,
                storage_path=storage_path,
                name="cycling_classifier",
                resume="AUTO",
                checkpoint_score_attr='combined_metrics',
                scheduler=pop_scheduler,
                reuse_actors=True,
                stop = {
                    "val_loss": 0.005,
                    "training_iteration": 100
                },
                verbose=2,
            )

            # Enhanced results analysis
            best_trial = analysis.get_best_trial("combined_metrics", "max", "last")
            best_config = best_trial.config
            best_checkpoint = analysis.get_best_checkpoint(best_trial, "combined_metrics", "max")

            # Log detailed results
            print(f"Best trial config: {best_config}")
            print(f"Best combined metrics: {best_trial.last_result['combined_metrics']}")
            print(f"Best validation loss: {best_trial.last_result['val_loss']}")
            print(f"Best validation F1: {best_trial.last_result['val_f1']}")
            print(f"Best validation AUROC: {best_trial.last_result['val_auroc']}")
            print(f"Best checkpoint path: {best_checkpoint}")

            return best_trial, best_config, best_checkpoint
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            raise
        finally:
            ray.shutdown()




def compare_capping_methods(df):
    # Original stats
    original_stats = df.agg(['min', 'max', 'mean', 'std']).round(3)
    
    # Different IQR multipliers
    iqr_multipliers = {
        'Conservative (2.0 IQR)': 2.0,
        'Standard (1.5 IQR)': 1.5,
        'Aggressive (1.0 IQR)': 1.0
    }
    
    # Z-score thresholds
    z_thresholds = {
        'Z-score (3 std)': 3,
        'Z-score (2.5 std)': 2.5,
        'Z-score (2 std)': 2
    }
    
    results = {}
    
    # IQR methods
    for name, multiplier in iqr_multipliers.items():
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df_capped = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
        results[name] = df_capped.agg(['min', 'max', 'mean', 'std']).round(3)
    
    # Z-score methods
    for name, threshold in z_thresholds.items():
        z_scores = stats.zscore(df)
        df_capped = df.copy()
        df_capped[abs(z_scores) > threshold] = df.clip(
            lower=df.mean() - threshold * df.std(),
            upper=df.mean() + threshold * df.std(),
            axis=1
        )[abs(z_scores) > threshold]
        results[name] = df_capped.agg(['min', 'max', 'mean', 'std']).round(3)
    
    # Calculate percentage of values capped for each method
    capped_percentages = {}
    for name, multiplier in iqr_multipliers.items():
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        capped = ((df < lower_bound) | (df > upper_bound)).sum().sum()
        capped_percentages[name] = (capped / (df.shape[0] * df.shape[1]) * 100).round(2)
    
    for name, threshold in z_thresholds.items():
        z_scores = stats.zscore(df)
        capped = (abs(z_scores) > threshold).sum().sum()
        capped_percentages[name] = (capped / (df.shape[0] * df.shape[1]) * 100).round(2)
    
    return results, capped_percentages


def cap_outliers_IQR(df, multiplier=1.5):
    # Create a copy of the dataframe
    df_capped = df.copy()
    
    for column in df.columns:
        # Calculate Q1, Q3, and IQR for each column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Only apply capping if IQR is not zero
        if IQR != 0:
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)            
        else:
            print(f"\n{column}: Skipped capping (no variation in quartiles)")
    
    return df_capped