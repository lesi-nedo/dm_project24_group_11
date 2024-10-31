import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from sklearn.svm import OneClassSVM
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerPathCollection
from typing import Tuple, Optional
from scipy.stats import randint, uniform, loguniform
import warnings

real_cpu_count = psutil.cpu_count(logical=False)

def visualize_one_class_svm(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced visualizations for OneClassSVM outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    anomaly_scores : numpy.ndarray
        Anomaly scores from the model (negative distance from hyperplane)
    mask : numpy.ndarray
        Boolean mask indicating outliers
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 7))
    gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Plot 1: PCA visualization with improved aesthetics
    ax1 = fig.add_subplot(gs[0])
    
    # Separate outliers and inliers
    outliers = df[mask]
    inliers = df[~mask]
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.values)
    X_outliers = pca.transform(outliers.values)
    X_inliers = pca.transform(inliers.values)
    
    # Create enhanced scatter plot
    ax1.scatter(X_inliers[:, 0], X_inliers[:, 1],
                c='royalblue', label='Inliers', alpha=0.6, s=50)
    ax1.scatter(X_outliers[:, 0], X_outliers[:, 1],
                c='crimson', label='Outliers', alpha=0.8, s=100)
    
    # Add explanatory text
    variance_ratio = pca.explained_variance_ratio_
    ax1.set_xlabel(f'First Principal Component\n({variance_ratio[0]:.1%} variance explained)')
    ax1.set_ylabel(f'Second Principal Component\n({variance_ratio[1]:.1%} variance explained)')
    ax1.set_title('PCA Projection of Data\nwith OneClassSVM Detection', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision Function visualization
    ax2 = fig.add_subplot(gs[1])
    
    # Normalize scores for visualization
    normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Create scatter plot with size proportional to anomaly score
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=anomaly_scores,
                         cmap='RdYlBu_r',
                         s=1000 * normalized_scores,
                         alpha=0.6,
                         label='Decision scores')
    
    # Customize the plot
    ax2.set_title('OneClassSVM Decision Scores\nMarker Size ∝ Anomaly Score', pad=20)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.grid(True, alpha=0.3)
    
    # Add legend with custom handler
    def update_legend_marker_size(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([100])
    
    ax2.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
    
    # Add colorbar
    norm = plt.Normalize(anomaly_scores.min(), anomaly_scores.max())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
    plt.colorbar(sm, ax=ax2, label='Decision Score', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print("\nOneClassSVM Detection Summary:")
    print(f"Total points: {len(df)}")
    print(f"Outliers detected: {mask.sum()} ({mask.sum()/len(df):.1%})")
    print(f"Decision Score Range: {anomaly_scores.min():.2f} to {anomaly_scores.max():.2f}")

def one_class_svm_scoring(estimator, X):
    """
    Custom scoring function for OneClassSVM.
    Combines decision boundary margin and prediction distribution.
    
    Parameters:
    -----------
    estimator : Pipeline
        The pipeline containing OneClassSVM
    X : array-like
        The input samples
    
    Returns:
    --------
    float
        Score value (higher is better)
    """
    # Get the OneClassSVM from pipeline
    svm = estimator.named_steps['svm']
    
    # Get decision function scores
    scores = svm.decision_function(X)
    
    # Calculate metrics
    margin = np.abs(scores).mean()  # Average distance from decision boundary
    score_std = np.std(scores)  # Higher std suggests better separation
    
    # Combine metrics (normalized margin and score standard deviation)
    normalized_margin = 1 / (1 + np.exp(-margin))  # Sigmoid to bound margin
    combined_score = (normalized_margin + score_std) / 2
    
    return combined_score

class OneClassSVMDetector:
    """
    Enhanced OneClassSVM-based outlier detector with visualization capabilities
    and hyperparameter search.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: str = 'scale',
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the OneClassSVM detector with enhanced parameters.
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            Kernel type to be used
        nu : float, default=0.1
            An upper bound on the fraction of training errors
        gamma : str or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        random_state : int or None, default=None
            Random state for reproducibility
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', OneClassSVM(
                kernel=kernel,
                nu=nu,
                gamma=gamma
            ))
        ])
        
        # Define parameter distributions for random search
        self.param_distributions = {
            'svm__nu': uniform(0.01, 0.49),  # nu must be in (0, 1)
            'svm__gamma': loguniform(1e-4, 1),  # log-uniform for gamma
            'svm__kernel': ['rbf', 'sigmoid'],  # most common kernels for outlier detection
        }
    
    def fit_predict_with_search(
        self,
        df: pd.DataFrame,
        plot_results: bool = True,
        n_iter: int = 20,
        cv: int = 5,
        verbose: int = 1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit and predict outliers using OneClassSVM with RandomizedSearchCV
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing features to detect outliers
        plot_results : bool, default=True
            Whether to visualize the results
        n_iter : int, default=20
            Number of iterations for RandomizedSearchCV
        cv : int, default=5
            Number of cross-validation folds
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        Tuple[pandas.DataFrame, numpy.ndarray]
            DataFrame containing outliers and array of anomaly scores
        """
        X = df.values
        warnings.filterwarnings("ignore")
        
        # Initialize and run randomized search
        search = RandomizedSearchCV(
            self.pipeline,
            self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=int(real_cpu_count * 0.2),
            verbose=verbose,
            scoring=make_scorer(one_class_svm_scoring)
        )
        
        search.fit(X)
        
        # Print best parameters and score
        print("\nBest parameters found:")
        for param, value in search.best_params_.items():
            print(f"{param}: {value}")
        print(f"Best cross-validation score: {search.best_score_:.3f}")
        
        # Use best estimator for predictions
        best_estimator = search.best_estimator_
        predictions = best_estimator.predict(X)
        self.mask = predictions == -1
        
        # Get anomaly scores (negative of decision function for consistency)
        anomaly_scores = -best_estimator.named_steps['svm'].decision_function(X)
        
        # Update pipeline with best estimator
        self.pipeline = best_estimator
        
        # Visualize results if requested
        if plot_results:
            visualize_one_class_svm(df, anomaly_scores, self.mask)
        
        warnings.filterwarnings("default")
        
        return df[self.mask], anomaly_scores
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the model and predict outliers using OneClassSVM.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing features to detect outliers
        plot_results : bool, default=True
            Whether to visualize the results
            
        Returns:
        --------
        Tuple[pandas.DataFrame, numpy.ndarray]
            DataFrame containing outliers and array of anomaly scores
        """
        # Fit and predict
        X = df.values
        predictions = self.pipeline.fit_predict(X)
        self.mask = predictions == -1
        
        # Get anomaly scores
        anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X)
        
        # Visualize results if requested
        if plot_results:
            visualize_one_class_svm(df, anomaly_scores, self.mask)
        
        return df[self.mask], anomaly_scores
    
    def get_search_results_summary(self, search_cv: RandomizedSearchCV) -> pd.DataFrame:
        """
        Get a summary of all trials from the hyperparameter search.
        
        Parameters:
        -----------
        search_cv : RandomizedSearchCV
            Completed RandomizedSearchCV object
            
        Returns:
        --------
        pandas.DataFrame
            Summary of all trials sorted by score
        """
        # Extract results
        results = []
        for params, score, rank in zip(
            search_cv.cv_results_['params'],
            search_cv.cv_results_['mean_test_score'],
            search_cv.cv_results_['rank_test_score']
        ):
            results.append({
                'rank': rank,
                'score': score,
                **{k.split('__')[1]: v for k, v in params.items()}
            })
        
        # Convert to DataFrame and sort by score
        results_df = pd.DataFrame(results).sort_values('score', ascending=False)
        return results_df

__all__ = ['OneClassSVMDetector']






# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import psutil
# import cupy as cp
# from cuml.svm import OneClassSVM as cuOCSVM
# from cuml.preprocessing import StandardScaler as cuStandardScaler
# from cuml.decomposition import PCA as cuPCA
# from typing import Tuple, Optional
# import warnings

# def gpu_one_class_svm_scoring(estimator, X):
#     """GPU-accelerated scoring function for OneClassSVM."""
#     svm = estimator.named_steps['svm']
#     scores = svm.decision_function(X)
#     scores_cpu = scores.get()  # Transfer to CPU for calculations
    
#     margin = np.abs(scores_cpu).mean()
#     score_std = np.std(scores_cpu)
    
#     normalized_margin = 1 / (1 + np.exp(-margin))
#     return (normalized_margin + score_std) / 2

# class GPUOneClassSVMDetector:
#     def __init__(
#         self,
#         kernel: str = 'rbf',
#         nu: float = 0.1,
#         gamma: str = 'scale',
#         random_state: Optional[int] = None
#     ) -> None:
#         self.pipeline = Pipeline([
#             ('scaler', cuStandardScaler()),
#             ('svm', cuOCSVM(
#                 kernel=kernel,
#                 nu=nu,
#                 gamma=gamma
#             ))
#         ])
        
#         self.param_distributions = {
#             'svm__nu': uniform(0.01, 0.49),
#             'svm__gamma': loguniform(1e-4, 1),
#             'svm__kernel': ['rbf', 'sigmoid'],
#         }

#     def fit_predict(
#         self,
#         df: pd.DataFrame,
#         plot_results: bool = True
#     ) -> Tuple[pd.DataFrame, np.ndarray]:
#         # Convert to GPU array
#         X = cp.array(df.values)
        
#         # Fit and predict on GPU
#         predictions = self.pipeline.fit_predict(X)
#         self.mask = predictions.get() == -1  # Transfer mask to CPU
        
#         # Get anomaly scores on GPU
#         anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X)
#         anomaly_scores_cpu = anomaly_scores.get()  # Transfer to CPU for visualization
        
#         if plot_results:
#             visualize_one_class_svm(df, anomaly_scores_cpu, self.mask)
        
#         return df[self.mask], anomaly_scores_cpu

#     def fit_predict_with_search(
#         self,
#         df: pd.DataFrame,
#         plot_results: bool = True,
#         n_iter: int = 20,
#         cv: int = 5,
#         verbose: int = 1
#     ) -> Tuple[pd.DataFrame, np.ndarray]:
#         X = cp.array(df.values)  # Convert to GPU array
        
#         search = RandomizedSearchCV(
#             self.pipeline,
#             self.param_distributions,
#             n_iter=n_iter,
#             cv=cv,
#             n_jobs=1,  # GPU doesn't benefit from multiple jobs
#             verbose=verbose,
#             scoring=make_scorer(gpu_one_class_svm_scoring)
#         )
        
#         search.fit(X)
        
#         print("\nBest parameters found:")
#         for param, value in search.best_params_.items():
#             print(f"{param}: {value}")
            
#         best_estimator = search.best_estimator_
#         predictions = best_estimator.predict(X)
#         self.mask = predictions.get() == -1
        
#         anomaly_scores = -best_estimator.named_steps['svm'].decision_function(X)
#         anomaly_scores_cpu = anomaly_scores.get()
        
#         self.pipeline = best_estimator
        
#         if plot_results:
#             visualize_one_class_svm(df, anomaly_scores_cpu, self.mask)
        
#         return df[self.mask], anomaly_scores_cpu