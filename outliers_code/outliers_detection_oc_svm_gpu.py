import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import torch

from thundersvm import OneClassSVM as thuOCSVM



from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
from typing import Tuple, List
from matplotlib.legend_handler import HandlerPathCollection
from scipy.stats import skew, kurtosis
from tqdm import tqdm

import psutil

# Use physical CPU cores for heavy computation
PHYSICAL_CORES = psutil.cpu_count(logical=False)



def update_legend_marker_size(handle, orig):
    """Customize size of the legend marker"""
    handle.update_from(orig)
    handle.set_sizes([100])

def visualize_one_class_svm_3d(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray) -> None:
    """Create enhanced 3D visualizations for OneClassSVM outlier detection results."""
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=(20, 8),  constrained_layout=True)
        gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Separate outliers and inliers
        outliers = df[mask]
        inliers = df[~mask]
        
        # Apply PCA with 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(df.values)
        
        # Check if we have both inliers and outliers
        has_inliers = len(inliers) > 0
        has_outliers = len(outliers) > 0
        
        # Plot 1: 3D PCA visualization
        ax1 = fig.add_subplot(gs[0], projection='3d')
        scatter_inliers = None
        scatter_outliers = None
        if has_inliers:
            X_inliers = pca.transform(inliers.values)
            scatter_inliers = ax1.scatter(X_inliers[:, 0], X_inliers[:, 1], X_inliers[:, 2],
                    c='royalblue', label='Inliers', alpha=0.6, s=50)
        
        if has_outliers:
            X_outliers = pca.transform(outliers.values)
            scatter_outliers = ax1.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2],
                    c='crimson', label='Outliers', alpha=0.8, s=100)
        
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_zlabel(f'PC3 ({variance_ratio[2]:.1%} var)')
        ax1.set_title('3D PCA Projection with OneClassSVM', pad=20)
        
        if has_inliers or has_outliers:
            ax1.legend(
                handler_map={
                    scatter_inliers: HandlerPathCollection(update_func=update_legend_marker_size),
                    scatter_outliers: HandlerPathCollection(update_func=update_legend_marker_size)
                }
            )
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

        # Set initial view angle
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score', alpha=0.7)
        
        # Add legend with custom handler
        ax2.legend(['Data Points'], handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
        
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
        


def get_gpu_id():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None
    
# Usage:
GPU_ID = get_gpu_id()

print(f"One-Class SVM Will Run ON GPU ID: {GPU_ID}")

# Use physical CPU cores for computation
PHYSICAL_CORES = psutil.cpu_count(logical=False)
THREAD_COUNT = psutil.cpu_count(logical=True)

@dataclass
class SVMParams:
    """Dataclass to store OneClassSVM parameters"""
    kernel: str
    nu: float
    gamma: float


def compute_gamma(params, X: np.ndarray) -> float:
        gamma = 0
        if params.gamma == 'auto':
            gamma = 1/X.shape[1]
        elif params.gamma == 'scale':
            gamma = (1/(X.shape[1] * X.var()))
        else:
            try:
                gamma = float(params.gamma)
            except:
                raise ValueError("Invalid gamma value. Use 'auto', 'scale', or a float value")
        return gamma

def gpu_svm_scoring(params: SVMParams, X: np.ndarray) -> Tuple[float, SVMParams]:
    """GPU version of OneClassSVM scoring with CUDA streams"""

    # Check for NaN/infinite values
    if np.any(~np.isfinite(X)):
        print("Warning: Input contains NaN or infinite values. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale X if values are too large
    max_val = np.abs(X).max()
    if max_val > 1e4:
        X = X / max_val
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', thuOCSVM(
            kernel=params.kernel,
            nu=params.nu,
            gamma=params.gamma,
            gpu_id=GPU_ID
        ))
    ])
    
    # Fit and score with error handling
    try:
        pipeline.fit(X)
        scores = np.array(pipeline.named_steps['svm'].decision_function(X).flatten())
        
        # Handle NaN/infinite scores
        scores = np.clip(scores, -1e10, 1e10)
        scores = np.nan_to_num(scores, nan=np.mean(scores[np.isfinite(scores)]))
        
        # Calculate metrics with numerical stability
        score_mean = np.nanmean(scores)
        score_std = np.nanstd(scores)
        
        valid_scores = scores[np.isfinite(scores)]
        
        if len(valid_scores) < 2:
            return 0.0, params
            
        score_range = np.ptp(valid_scores)

        valid_mean = np.mean(valid_scores)
        valid_std = np.std(valid_scores)

        eps = 1e-10
        if valid_std < eps:
            valid_std = eps
        
        if score_range < 1e-7:
            score_skew = 0.0
            score_kurtosis = 0.0
        else:
            normalized_scores = (valid_scores - valid_mean) / (valid_std + eps)
            normalized_scores = np.clip(normalized_scores, -10, 10)

            score_skew = float(abs(skew(normalized_scores, bias=False)))
            score_kurtosis = float(kurtosis(normalized_scores, bias=False))
        
        # Rest of the calculations with nan handling
        extreme_threshold = score_mean - (3 * score_std)
        extreme_ratio = float(np.sum(np.isfinite(scores) & (scores < extreme_threshold))) / len(scores)
        extreme_penalty = np.exp(extreme_ratio * 5) if extreme_ratio > 0.01 else 0.5
        
        # Use nanpercentile for percentile calculations
        # With:
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) > 0:
            percentiles = np.percentile(finite_scores, [0.1, 1, 5, 95, 99, 99.9])
        else:
            # Return default values if no valid scores
            return 0.0, params

        # Rest of the code remains the same
        density_gaps = np.diff(percentiles)
        tail_separation = float(density_gaps[0] + density_gaps[-1])
        density_score = float(np.nanmean(density_gaps[1:-1])) / (tail_separation + 1e-10)
        
        threshold_factor = 5.0 + score_skew + (0.1 * score_kurtosis) + (2.0 * extreme_ratio)
        threshold_score = score_mean - (threshold_factor * score_std)
        potential_outliers = float(np.sum(np.isfinite(scores) & (scores < threshold_score))) / len(scores)
        outlier_penalty = np.exp(potential_outliers * 4) if potential_outliers > 0.01 else 0.5
        
        combined_score = (
            0.25 * float(score_std) +
            0.2 * tail_separation +
            0.2 * (1 / (1 + score_skew)) +
            0.1 * (1 / float(outlier_penalty)) +
            0.15 * density_score +
            0.1 * (1 / float(extreme_penalty))
        )
        
        nu_penalty = np.exp(params.nu * 15)
        combined_score = combined_score / float(nu_penalty)
        
        return 1 / (1 + np.exp(-combined_score)), params
        
    except Exception as e:
        print(f"Warning: Error in scoring: {e}")
        return 0.0, params


class GPUOneClassSVMDetector:
    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: float = 0.1
    ) -> None:
        """Initialize GPU-accelerated One-Class SVM detector with enhanced pipeline.
        
        Args:
            kernel: Kernel type ('rbf' or 'sigmoid')
            nu: An upper bound on the fraction of training errors
            gamma: Kernel coefficient
        """
        self.nu = min(max(nu, 0.01), 0.5)
        
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        
    
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

    def _print_results(self, df: pd.DataFrame, best_params: SVMParams, best_score: float, anomaly_scores: np.ndarray):
        """Print formatted results of OneClassSVM detection."""
        print("\n" + "="*50)
        print("GPU One-Class SVM Detection Results")
        print("="*50)
        
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

    def _generate_param_combinations(self, n_iter: int) -> List[SVMParams]:
        """Generate parameter combinations optimized for ~1% outlier detection.
        
        Args:
            n_iter: Number of parameter combinations to generate
                
        Returns:
            List of parameter combinations
        """
        params_list = []
        for _ in range(n_iter):
            # Set nu to target roughly 1% outliers with some variation
            # Using a narrower range around 0.01 (1%)
            nu = np.random.uniform(0.005, 0.015)
            
            # Adjust gamma range for more conservative boundary estimates
            # Using log scale but with a narrower range focused on smaller values
            gamma_exp = np.random.uniform(np.log(1e-5), np.log(1))
            gamma = str(np.exp(gamma_exp))
            
            # Prefer 'rbf' kernel as it typically provides better control over decision boundaries
            kernel = np.random.choice(['rbf', 'sigmoid'], p=[0.8, 0.2])
            
            params = SVMParams(
                kernel=kernel,
                nu=nu,
                gamma=gamma  # Remove 'auto' and 'scale' options for better control
            )
            params_list.append(params)
        
        return params_list

    def _adaptive_nu(self, scores: np.ndarray) -> float:
        """Adaptively determine nu parameter based on score distribution.
        
        Args:
            scores: Array of decision function scores
            
        Returns:
            float: Adapted nu value within valid range
        """
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        threshold = q1 - 1.5 * iqr
        
        nu = np.mean(scores < threshold)
        return min(max(nu, 0.01), 0.5)
    
    

    def fit_predict(
    self,
    df: pd.DataFrame,
    plot_results: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit the model and predict outliers using GPU acceleration."""
        X = df.values
        gamma = compute_gamma(self, X)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', thuOCSVM(
                kernel=self.kernel,
                nu=self.nu,
                gamma=gamma,
                gpu_id = GPU_ID
            ))
        ])
        
        gamma = min(max(gamma, 0.001), 100.0)
        # Fit first, then predict
        self.pipeline.fit(X)
        predictions = self.pipeline.predict(X)  # This returns numpy array
        
        self.mask = np.array(predictions == -1)
        
        anomaly_scores = -self.pipeline.named_steps['svm'].decision_function(X).flatten()
        
        if plot_results:
            self._print_results(df, self.pipeline.named_steps['svm'].get_params(), 0.0, anomaly_scores)
            self._plot(df, anomaly_scores)
            
        return df[self.mask], anomaly_scores

    def fit_predict_with_search(
        self,
        df: pd.DataFrame,
        n_iter: int = 20,
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """GPU-accelerated parameter search using CUDA streams"""
        
        # Convert data to GPU once
        X = np.array(df.values)
        params_list = self._generate_param_combinations(n_iter)
        
    
        
        # Run parameter search on GPU
        results = []
        for i, params in tqdm(enumerate(params_list), desc="Parameter Search", total=len(params_list)):
            params.gamma = compute_gamma(params, X)
            print(f"Processing params: {params}")
            score, params = gpu_svm_scoring(params, X)
            results.append((score, params))
            
        
        # Find best parameters
        best_score, best_params = max(results, key=lambda x: x[0])
        
        # Configure final model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', thuOCSVM(
                kernel=best_params.kernel,
                nu=best_params.nu,
                gamma=best_params.gamma,
                gpu_id=GPU_ID
            ))
        ])
        
        # Final fit and predict
        self.pipeline.fit(X)
        predictions = self.pipeline.predict(X) 

        self.mask = np.array(predictions == -1)
        anomaly_scores = self.pipeline.named_steps['svm'].decision_function(X).flatten()
        
        if plot_results:
            self._print_results(df, best_params.__dict__, best_score, anomaly_scores)
            self._plot(df, anomaly_scores)
        
        return df[self.mask], anomaly_scores 


__all__ = ['GPUOneClassSVMDetector']

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=1, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    
    detector = GPUOneClassSVMDetector(nu=0.48, kernel='rbf', gamma='scale')
    outliers, scores = detector.fit_predict_with_search(df)
    detector.fit_predict(df)
    # print(f"Found {len(outliers)} outliers")