import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import cupy as cp
from cuml.svm import OneClassSVM as cuOCSVM
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.decomposition import PCA as cuPCA
from typing import Tuple, Optional
import warnings

def gpu_one_class_svm_scoring(estimator, X):
    """GPU-accelerated scoring function for OneClassSVM."""
    svm = estimator.named_steps['svm']
    scores = svm.decision_function(X)
    scores_cpu = scores.get()  # Transfer to CPU for calculations
    
    margin = np.abs(scores_cpu).mean()
    score_std = np.std(scores_cpu)
    
    normalized_margin = 1 / (1 + np.exp(-margin))
    return (normalized_margin + score_std) / 2

class GPUOneClassSVMDetector:
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: str = 'scale',
        random_state: Optional[int] = None
    ) -> None:
        self.pipeline = Pipeline([
            ('scaler', cuStandardScaler()),
            ('svm', cuOCSVM(
                kernel=kernel,
                nu=nu,
                gamma=gamma
            ))
        ])
        
        self.param_distributions = {
            'svm__nu': uniform(0.01, 0.49),
            'svm__gamma': loguniform(1e-4, 1),
            'svm__kernel': ['rbf', 'sigmoid'],
        }

    def fit_predict(
        self,
        df: pd.DataFrame,
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        # Convert to GPU array
        X = cp.array(df.values)
        
        # Fit and predict on GPU
        predictions = self.pipeline.fit_predict(X)
        self.mask = predictions.get() == -1  # Transfer mask to CPU
        
        # Get anomaly scores on GPU
        anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X)
        anomaly_scores_cpu = anomaly_scores.get()  # Transfer to CPU for visualization
        
        if plot_results:
            visualize_one_class_svm(df, anomaly_scores_cpu, self.mask)
        
        return df[self.mask], anomaly_scores_cpu

    def fit_predict_with_search(
        self,
        df: pd.DataFrame,
        plot_results: bool = True,
        n_iter: int = 20,
        cv: int = 5,
        verbose: int = 1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        X = cp.array(df.values)  # Convert to GPU array
        
        search = RandomizedSearchCV(
            self.pipeline,
            self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=1,  # GPU doesn't benefit from multiple jobs
            verbose=verbose,
            scoring=make_scorer(gpu_one_class_svm_scoring)
        )
        
        search.fit(X)
        
        print("\nBest parameters found:")
        for param, value in search.best_params_.items():
            print(f"{param}: {value}")
            
        best_estimator = search.best_estimator_
        predictions = best_estimator.predict(X)
        self.mask = predictions.get() == -1
        
        anomaly_scores = -best_estimator.named_steps['svm'].decision_function(X)
        anomaly_scores_cpu = anomaly_scores.get()
        
        self.pipeline = best_estimator
        
        if plot_results:
            visualize_one_class_svm(df, anomaly_scores_cpu, self.mask)
        
        return df[self.mask], anomaly_scores_cpu