import json
import pickle
import tempfile
import lightning as pl
import ray
import ray.air
import ray.train
import ray.tune
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from functools import partial

from .Autoencoder import Autoencoder
from .logging import setup_logger

logger = setup_logger("Autoencoder")


DEFAULT_ROOT_DIR = "models"



class GenerateCallback(Callback):
    def __init__(self, tuple_data, every_n_epochs=1):
        super().__init__()
        data_input, self.feature_names = tuple_data
        self.sample_data = data_input
        self.data_input = None
        self.every_n_epochs = every_n_epochs

    def setup(self, trainer, pl_module, stage):
        self.data_input = self.sample_data.to(pl_module.device)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct features
            with torch.no_grad():
                pl_module.eval()
                reconst_features = pl_module(self.data_input)
                pl_module.train()

            # Convert to numpy for plotting
            error = pl_module.loss_function(reconst_features, self.data_input)
            input_np = self.data_input.cpu().numpy()
            reconst_np = reconst_features.cpu().numpy()
            print(f"Epoch {trainer.current_epoch}: Loss: {error.item()}")
            self.feature_names = [f"Feature_{i}" for i in range(input_np.shape[1])]

            # Create comparison plots for each sample
        
            fig = plt.figure(figsize=(12, 4))
            
            # Original features
            plt.subplot(1, 2, 1)
            plt.bar(self.feature_names, input_np.mean(axis=0))
            plt.title('Original Features')
            plt.xticks(rotation=45)
            
            # Reconstructed features
            plt.subplot(1, 2, 2)
            plt.bar(self.feature_names, reconst_np.mean(axis=0))
            plt.title('Reconstructed Features')
            plt.xticks(rotation=45)
            
            plt.tight_layout()

            
            # Log to tensorboard
            trainer.logger.experiment.add_figure(
                f"Reconstructions/Sample_{trainer.current_epoch}", 
                fig, 
                global_step=trainer.global_step
            )
            plt.close(fig)

            errors = torch.mean((reconst_features - self.data_input) ** 2, dim=0) 
            # Also log the mean reconstruction error for each feature
            for feat_idx, feat_name in enumerate(self.feature_names):
                trainer.logger.experiment.add_scalar(
                    f"Reconstruction_Error/{feat_name}",
                    errors[feat_idx],
                    global_step=trainer.global_step
                )



class OurDataset(Dataset):
    def __init__(self, csv_file:str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.label = self.data.loc[:, "label"]
        self.data = self.data.drop(columns=["label"])
        self._change_to_numeric()

        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled_data, columns=self.data.columns)
        self.feature_names = self.data.columns
        self.shape = self.data.shape
        self.data_tensor = torch.tensor(self.data.values, dtype=torch.float32)
        self.label_tensor = torch.tensor(self.label.values, dtype=torch.float32)
        self.transform = transform

    def _change_to_numeric(self):
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce", downcast="float")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        single_race_tensor = self.data_tensor[idx, :]
        single_label_tensor = self.label_tensor[idx]
        

        sample = {
            "race": single_race_tensor,
            "label": single_label_tensor
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
    def to_tensor(self):
        return self.data_tensor
    
def get_data_train(num_samples: int, dataset: Dataset):
    sub_set = torch.stack([dataset[i] for i in range(num_samples)], dim=0)
    return sub_set

def train_autoencoder(
        configs: dict, dataset_path: str, optimizer_configs:dict = None, 
        lr_scheduler_configs:dict = None, max_epochs: int = 10, 
        num_folds: int = 5, monitor: str = "val_loss", noise: np.ndarray = None, noise_level: float = 0.0001, 
        trainer_config: dict = None    
):
    

    latent_dim = configs["latent_dim"]
    scale_factor = configs["scale_factor"]
    act_fn = configs["act_fn"]
    dropout = configs.get("dropout", None)
    batch_size = configs["batch_size"]
    optimizer = configs["optimizer"]
    lr_scheduler = configs["lr_scheduler"]

    if configs.get("weight_decay", None) is not None:
        optimizer_configs["weight_decay"] = configs["weight_decay"]
    
    if configs.get("lr", None) is not None:
        optimizer_configs["lr"] = configs["lr"]

    num_hidden_layers = len(scale_factor)
    data_input = OurDataset(dataset_path)

    torch.set_float32_matmul_precision('medium')


    
    model_save_dir = os.path.join(DEFAULT_ROOT_DIR, "autoencoder")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"autoencoder-{latent_dim}-{num_hidden_layers}-{timestamp}"
    
    # Create KFold object with stratification based on data distribution
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    # Track metrics across folds
    fold_metrics = {
        'train_losses': [],
        'val_losses': [],
        'reconstruction_errors': []
    }


    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_input.to_tensor())):
        
        X_train = data_input[train_idx]["race"]
        X_val = data_input[val_idx]["race"]
        
        train_loader = DataLoader(X_train, batch_size=batch_size, num_workers=5, shuffle=True)
        val_loader = DataLoader(X_val, batch_size=batch_size, num_workers=5, shuffle=False)
        
        
        logger_tensor = TensorBoardLogger(
            save_dir=os.path.join(DEFAULT_ROOT_DIR, "ae_logs"),
            name=f"fold_{fold+1}",
            version=timestamp
        )

        # Add fold-specific callbacks
        callbacks = [
            # ModelCheckpoint(
            #     monitor=monitor,
            #     dirpath=fold_dir,
            #     filename=f"{file_name}-fold{fold+1}",
            #     save_top_k=1,
            #     mode="min",
            # ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor=monitor,
                patience=5,
                mode="min",
                verbose=False
            ),
        ]

        autoencoder = Autoencoder(
            data_input.shape[1], latent_dim,
            scale_factor, act_fn, optimizer, optimizer_configs, dropout,
            lr_scheduler, lr_scheduler_configs, monitor,
            noise, noise_level
        )


        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger_tensor,
            callbacks=callbacks,
            **trainer_config
        )

       # trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None
        steps_per_epoch = len(data_input) // batch_size


        if str(lr_scheduler) == str(OneCycleLR):
            lr_scheduler_configs.update({
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * max_epochs,
            "max_lr": optimizer_configs["lr"] * 10
        })

        
        trainer.fit(autoencoder, train_loader, val_loader)

        # Collect fold metrics
        
        val_loss = trainer.callback_metrics['val_loss'].item()
        train_loss = trainer.callback_metrics['train_loss'].item()
        mse_reconstruction_error = trainer.callback_metrics['mse_recon_loss'].item()

        
        fold_metrics['train_losses'].append(train_loss)
        fold_metrics['val_losses'].append(val_loss)
        fold_metrics['reconstruction_errors'].append(mse_reconstruction_error)


    # Calculate and log cross-validation statistics
    mean_val_loss = np.mean(fold_metrics['val_losses'])
    std_val_loss = np.std(fold_metrics['val_losses'])
    mean_reconstruction_error = np.mean(fold_metrics['reconstruction_errors'])
    std_val_loss = np.std(fold_metrics['reconstruction_errors'])
    mean_train_loss = np.mean(fold_metrics['train_losses'])
    std_train_loss = np.std(fold_metrics['train_losses'])
    

    final_metrics = {
        "val_loss": mean_val_loss,
        "reconstruction_error": mean_reconstruction_error,
        "combined_metric": mean_val_loss + 0.5 * mean_reconstruction_error
    }
    
    # Create temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:

        
        # Save metrics
        metrics_path = os.path.join(temp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "mean_train_loss": mean_train_loss,
                "std_train_loss": std_train_loss,
                "mean_val_loss": mean_val_loss,
                "std_val_loss": std_val_loss,
                "mean_reconstruction_error": mean_reconstruction_error,
                "std_reconstruction_error": std_val_loss
            }, f)
        
        # Create checkpoint from directory
        checkpoint = Checkpoint.from_directory(temp_dir)

        train.report(metrics=final_metrics, checkpoint=checkpoint)
            
    return final_metrics   
    
def train_wrapper_autoencoder(configs: dict, args_dict: dict):
    optimizer = configs["optimizer"]
    lr_scheduler = configs["lr_scheduler"]
    optimizer_configs = args_dict["optimizers_configs"][str(optimizer)]
    lr_scheduler_configs = args_dict["lr_schedulers_configs"][str(lr_scheduler)]

    
    args_dict_copy = args_dict.copy()
    args_dict_copy.pop("optimizers_configs")
    args_dict_copy.pop("lr_schedulers_configs")
    args_dict_copy["optimizer_configs"] = optimizer_configs
    args_dict_copy["lr_scheduler_configs"] = lr_scheduler_configs
    return train_autoencoder(configs, **args_dict_copy)


def main_autoencoder(num_samples=25, max_num_epochs=50):
    # Enhanced search space with more granular options
    configs = {
        "latent_dim": tune.choice([2, 3]),
        "scale_factor": tune.choice([
            [0.5], 
            [0.7, 0.5],
            [0.8, 0.6]
        ]),
        "act_fn": tune.choice(["GELU", "LeakyReLU", "Tanhshrink", "SELU","ReLU"]),
        "dropout": tune.uniform(0.1, 0.5),  # More granular dropout range
        "batch_size": tune.choice([32, 64, 128, 256]),
        "lr": tune.loguniform(1e-5, 1e-2),  # Adjusted learning rate range
        "optimizer": tune.choice([
            torch.optim.AdamW,  # Added AdamW
            torch.optim.SGD
        ]),
        "lr_scheduler": tune.choice([
            ReduceLROnPlateau,
        ]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
    }

    # Improved ASHA scheduler
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,  # Increased grace period
        reduction_factor=3,
        brackets=3
    )

    # Extended configuration dictionary
    configs_part = {
        "optimizers_configs": {
            str(torch.optim.Adam): {
                "weight_decay": None,
                "lr": None,
                "betas": (0.9, 0.999)
            },
            str(torch.optim.AdamW): {
                "weight_decay": None,
                "lr": None,
                "betas": (0.9, 0.999)
            },
            str(torch.optim.SGD): {
                "weight_decay": None,
                "lr": None,
                "momentum": 0.9,
                "nesterov": True
            }
        },
        "lr_schedulers_configs": {
            str(ReduceLROnPlateau): {
                "factor": 0.2,
                "patience": 5,  # Increased patience
                "threshold": 0.001,
                "min_lr": 1e-7,
                "mode": "min"
            },
            str(OneCycleLR): {
                "max_lr": None,  # Will be set based on lr
                "steps_per_epoch": None,  # Set in training
                "epochs": max_num_epochs,
                "pct_start": 0.3,
                "div_factor": 25.0
            }
        },
        "dataset_path": os.path.join(os.getcwd(), "DatasetClassification/TrainDataset.csv"),
        "max_epochs": max_num_epochs,
        "num_folds": 5,
        "monitor": "val_loss",
        "noise": None,
        "noise_level": 0.00001
    }

    # Setup paths with timestamps
    storage_path = os.path.abspath(os.path.join(os.getcwd(), "models", "ray_autoencoder"))
    os.makedirs(storage_path, exist_ok=True)

    # Configure Ray with GPU
    ray.init(
        num_cpus=32,
        num_gpus=1,
        _system_config={
            "automatic_object_spilling_enabled": True,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}
            ),
        }
    )

    # Enhanced resource configuration
    resources_per_trial = {
        "cpu": 4,  # 32 CPUs / 8 parallel trials
        "gpu": 0.125  # Allow 8 trials to share 1 GPU
    }

    # Add GPU-specific configurations
    configs_part.update({
        "trainer_config": {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "precision": "16-mixed",  # Enable mixed precision training
            "enable_progress_bar": False,
            # "gradient_clip_val": 1.0,

        }

    })

    # Enhanced Ray Tune run configuration
    result = tune.run(
        partial(train_wrapper_autoencoder, args_dict=configs_part),
        resources_per_trial=resources_per_trial,
        config=configs,
        num_samples=num_samples,
        scheduler=scheduler,
        # search_alg=search_alg,
        storage_path=storage_path,
        keep_checkpoints_num=3,
        checkpoint_score_attr="mean-val_loss",
        stop={
            "val_loss": 0.001,
            "training_iteration": max_num_epochs
        },
        # resume=True,
        verbose=1,
    )

    # Enhanced results analysis
    best_trial = result.get_best_trial("combined_metric", "min", "last")
    best_config = best_trial.config
    best_checkpoint = result.get_best_checkpoint(best_trial, "combined_metric", "min")

    # Log detailed results
    print(f"Best trial config: {best_config}")
    print(f"Best validation loss: {best_trial.last_result['combined_metric']}")
    print(f"Best checkpoint path: {best_checkpoint}")

    return best_trial, best_config, best_checkpoint
    