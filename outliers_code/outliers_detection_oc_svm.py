import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerPathCollection
from scipy.stats import uniform, loguniform, skew, kurtosis

# Number of physical CPU cores
PHYSICAL_CORES = psutil.cpu_count(logical=False)

def update_legend_marker_size(handle, orig):
    """Customize size of the legend marker"""
    handle.update_from(orig)
    handle.set_sizes([100])

def visualize_one_class_svm_3d(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced 3D visualizations for OneClassSVM outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    anomaly_scores : numpy.ndarray
        Anomaly scores from the model (negative distance from hyperplane)
    mask : numpy.ndarray
        Boolean mask indicating outliers
    """
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=(20, 8), constrained_layout=True)
        gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Separate outliers and inliers
        outliers = df[mask]
        inliers = df[~mask]
        
        # Apply PCA with 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(df.values)
        X_outliers = pca.transform(outliers.values)
        X_inliers = pca.transform(inliers.values)
        
        # Plot 1: 3D PCA visualization
        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax1.scatter(X_inliers[:, 0], X_inliers[:, 1], X_inliers[:, 2],
                    c='royalblue', label='Inliers', alpha=0.6, s=50)
        ax1.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2],
                    c='crimson', label='Outliers', alpha=0.8, s=100)
        
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_zlabel(f'PC3 ({variance_ratio[2]:.1%} var)')
        ax1.set_title('3D PCA Projection with OneClassSVM', pad=20)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 3D Anomaly Score visualization
        ax2 = fig.add_subplot(gs[1], projection='3d')
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                            c=anomaly_scores,
                            cmap='RdYlBu_r',
                            s=100,
                            alpha=0.6)
        
        # Highlight high anomaly scores
        high_anomaly_mask = anomaly_scores > np.percentile(anomaly_scores, 90)
        if np.any(high_anomaly_mask):
            ax2.scatter(X_pca[high_anomaly_mask, 0],
                    X_pca[high_anomaly_mask, 1],
                    X_pca[high_anomaly_mask, 2],
                    s=200,
                    facecolors='none',
                    edgecolors='red',
                    alpha=0.5,
                    label='High Anomaly Score')
        
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title('3D Anomaly Score Distribution', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score', alpha=0.7)
        
        # Add legend with custom handler
        ax2.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
        
        plt.show()
        
        # Print summary statistics
        print(f"Variance explained by 3 PCs: {variance_ratio.sum():.1%}")
        print("\nInteractive Controls:")
        print("- Use arrow keys to rotate and tilt the plots")
        print("- Close the plot window to exit")

def visualize_outliers_2d(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    mask: np.ndarray,
    fig_size: Tuple[int, int] = (15, 7)
) -> None:
    """Optimized 2D visualization of outliers"""
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Perform PCA once
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df.values)
        
        # Plot 1: PCA projection
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(X_pca[~mask, 0], X_pca[~mask, 1],
                   c='royalblue', label='Inliers', alpha=0.6, s=50)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c='crimson', label='Outliers', alpha=0.8, s=100)
        
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_title('OneClassSVM Outlier Detection')
        ax1.legend()
        
        # Plot 2: Decision scores
        ax2 = fig.add_subplot(gs[1])
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=anomaly_scores,
                            cmap='RdYlBu_r',
                            s=100,
                            alpha=0.6)
        
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Decision Score Distribution')
        
        plt.show()

    
        print(f"Variance explained by 2 PCs: {sum(variance_ratio):.1%}")


# Use physical CPU cores for computation
PHYSICAL_CORES = psutil.cpu_count(logical=False)
THREAD_COUNT = psutil.cpu_count(logical=True)

@dataclass
class SVMParams:
    """Dataclass to store OneClassSVM parameters"""
    kernel: str
    nu: float
    gamma: float

def parallel_svm_scoring(params: SVMParams, X: np.ndarray) -> Tuple[float, SVMParams]:
    """Parallel version of OneClassSVM scoring with improved metrics"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', OneClassSVM(
            kernel=params.kernel,
            nu=params.nu,
            gamma=params.gamma
        ))
    ])
    
    # Fit SVM and get scores
    pipeline.fit(X)
    scores = -pipeline.named_steps['svm'].decision_function(X)
    
     # Core metrics
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_skew = np.abs(skew(scores))
    score_kurtosis = kurtosis(scores)
    
    # Extreme value analysis - now used
    extreme_threshold = score_mean - (3 * score_std)
    extreme_ratio = np.sum(scores < extreme_threshold) / len(scores)
    extreme_penalty = np.exp(extreme_ratio * 5) if extreme_ratio > 0.01 else 0.5
    
    # Density metrics
    percentiles = np.percentile(scores, [0.1, 1, 5, 95, 99, 99.9])
    density_gaps = np.diff(percentiles)
    tail_separation = density_gaps[0] + density_gaps[-1]
    density_score = np.mean(density_gaps[1:-1]) / (density_gaps[0] + density_gaps[-1])
    
    # Adaptive threshold using extreme ratio
    threshold_factor = 5.0 + score_skew + (0.1 * score_kurtosis) + (2.0 * extreme_ratio)
    threshold_score = score_mean - (threshold_factor * score_std)
    potential_outliers = np.sum(scores < threshold_score) / len(scores)
    outlier_penalty = np.exp(potential_outliers * 4) if potential_outliers > 0.01 else 0.5
    
    # Combined score with extreme ratio component
    combined_score = (
        0.25 * score_std +                    # Distribution spread
        0.2 * tail_separation +              # Tail behavior
        0.2 * (1 / (1 + score_skew)) +      # Symmetry
        0.1 * (1 / outlier_penalty) +        # Outlier penalty
        0.15 * density_score +                # Density separation
        0.1 * (1 / extreme_penalty)          # Extreme value control
    )
    
    # Strong nu penalty
    nu_penalty = np.exp(params.nu * 15)
    combined_score = combined_score / nu_penalty
    
    return 1 / (1 + np.exp(-combined_score)), params

class OneClassSVMDetector:
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', OneClassSVM(
                kernel=kernel,
                nu=nu,
                gamma=gamma,
            ))
        ])
        
        self.param_distributions = {
            'nu': uniform(0.001, 0.1),
            'gamma': loguniform(1e-3, 1),
            'kernel': ['rbf', 'sigmoid']
        }

    def _generate_param_combinations(self, n_iter: int) -> list:
        """Generate parameter combinations for parallel processing"""
        params_list = []
        for _ in range(n_iter):
            params = SVMParams(
                kernel=np.random.choice(self.param_distributions['kernel']),
                nu=self.param_distributions['nu'].rvs(),
                gamma=self.param_distributions['gamma'].rvs()
            )
            params_list.append(params)
        return params_list

    def _plot(self, df: pd.DataFrame, anomaly_scores: np.ndarray):
        if df.shape[1] > 2:
            visualize_one_class_svm_3d(
                df,
                anomaly_scores,
                self.mask
            )
        else:
            visualize_outliers_2d(
                df,
                anomaly_scores,
                self.mask
            )
    def _print_results(self, df: pd.DataFrame, anomaly_scores: np.ndarray, best_params: SVMParams = None, best_score: float = None,):
        """Print formatted results of OneClassSVM detection."""
        print("\n" + "="*50)
        print("CPU One-Class SVM Detection Results")
        print("="*50)
        
        if best_params:
            # Model Configuration
            print("\nBest Parameters:")
            print(f"  Kernel:     {best_params['kernel']}")
            print(f"  Nu:         {best_params['nu']:.3f}")
            print(f"  Gamma:      {best_params['gamma']:.4f}")
            print(f"  Score:      {best_score:.4f}")
        
        # Detection Statistics
        n_outliers = self.mask.sum()
        outlier_ratio = n_outliers/len(df)
        print("\nDetection Statistics:")
        print(f"  Total points:      {len(df):,}")
        print(f"  Outliers found:    {n_outliers:,} ({outlier_ratio:.1%})")
        print(f"  Inliers retained:  {len(df)-n_outliers:,} ({1-outlier_ratio:.1%})")
        
        # Score Distribution
        print("\nAnomaly Score Distribution:")
        print(f"  Min score:     {anomaly_scores.min():.2f}")
        print(f"  Max score:     {anomaly_scores.max():.2f}")
        print(f"  Mean score:    {np.mean(anomaly_scores):.2f}")
        print(f"  Median score:  {np.median(anomaly_scores):.2f}")
        
        # Percentiles
        percentiles = np.percentile(anomaly_scores, [25, 75, 90, 95, 99])
        print("\nScore Percentiles:")
        print(f"  25th: {percentiles[0]:.2f}")
        print(f"  75th: {percentiles[1]:.2f}")
        print(f"  90th: {percentiles[2]:.2f}")
        print(f"  95th: {percentiles[3]:.2f}")
        print(f"  99th: {percentiles[4]:.2f}")
        
        print("\nVisualization Controls:")
        print("  - Use mouse to rotate 3D plots")
        print("  - Scroll to zoom in/out")
        print("  - Right-click and drag to pan")
        print("="*50 + "\n")

    def fit_predict_with_search(
        self, 
        df: pd.DataFrame, 
        n_iter: int = 20, 
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Optimized parallel parameter search"""
        X = df.values
        params_list = self._generate_param_combinations(n_iter)
        
        # Use ProcessPoolExecutor for CPU-bound parameter search
        with ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as executor:
            results = list(executor.map(
                partial(parallel_svm_scoring, X=X),
                params_list
            ))
        
        # Find best parameters
        best_score, best_params = max(results, key=lambda x: x[0])
        
        # Configure pipeline with best parameters
        self.pipeline.named_steps['svm'].set_params(
            kernel=best_params.kernel,
            nu=best_params.nu,
            gamma=best_params.gamma
        )
        
        # Final fit and predict
        predictions = self.pipeline.fit_predict(X)
        self.mask = predictions == -1
        anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X)
        
        if plot_results:
            self._print_results(df, anomaly_scores, self.pipeline.named_steps['svm'].get_params(), 0.0)
            self._plot(df, anomaly_scores)
        
        return df[self.mask], anomaly_scores

    def fit_predict(
        self, 
        df: pd.DataFrame, 
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Optimized basic fit and predict"""
        X = df.values
        predictions = self.pipeline.fit_predict(X)
        self.mask = predictions == -1
        anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X)
        if plot_results:
            self._plot(df, anomaly_scores)
        
        return df[self.mask], anomaly_scores




__all__ = ['OneClassSVMDetector']

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=1, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    
    detector = OneClassSVMDetector(nu=0.1, kernel='rbf', gamma='scale')
    outliers, scores = detector.fit_predict_with_search(df)
    detector.fit_predict(df)
    # print(f"Found {len(outliers)} outliers")
