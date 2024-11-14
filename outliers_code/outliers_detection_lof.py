import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple, Optional
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from scipy.stats import randint


real_cpu_count = psutil.cpu_count(logical=False)    # Number of physical cores

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


def visualize_outliers_3d(df: pd.DataFrame, lof_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced 3D visualizations for outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    lof_scores : numpy.ndarray
        LOF scores from the model
    mask : numpy.ndarray
        Boolean mask indicating outliers
    """
    # Set the style
    with plt.style.context('seaborn-v0_8-whitegrid'):
    
        print("\nInteractive Controls:")
        print("- Use arrow keys to rotate and tilt the plots")
        print("- Close the plot window to exit")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8), constrained_layout=True)
        
        # Separate outliers and inliers
        outliers = df[mask]
        inliers = df[~mask]
        
        # Apply PCA with 3 components
        pca = PCA(n_components=3)
        X = df.values
        X_pca = pca.fit_transform(X)
        X_outliers = pca.transform(outliers.values)
        X_inliers = pca.transform(inliers.values)
        
        # Plot 1: 3D PCA visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot inliers
        scatter1 = ax1.scatter(X_inliers[:, 0], X_inliers[:, 1], X_inliers[:, 2],
                            c='royalblue', label='Inliers', alpha=0.6, s=50)
        
        # Plot outliers
        scatter2 = ax1.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2],
                            c='crimson', label='Outliers', alpha=0.8, s=100)
        
        # Customize the plot
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_zlabel(f'PC3 ({variance_ratio[2]:.1%} var)')
        ax1.set_title('3D PCA Projection with Outliers', pad=20)
        ax1.legend()
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 3D LOF visualization
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Create color mapping based on LOF scores
        norm = plt.Normalize(lof_scores.min(), lof_scores.max())
        colors = plt.cm.RdYlBu_r(norm(lof_scores))
        
        # Plot points with color based on LOF score
        scatter3 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                            c=lof_scores,
                            cmap='RdYlBu_r',
                            s=100,
                            alpha=0.6)
        
        # Add spheres for high LOF scores
        high_lof_mask = lof_scores > np.percentile(lof_scores, 90)  # Top 10% of LOF scores
        if np.any(high_lof_mask):
            ax2.scatter(X_pca[high_lof_mask, 0],
                    X_pca[high_lof_mask, 1],
                    X_pca[high_lof_mask, 2],
                    s=200,
                    facecolors='none',
                    edgecolors='red',
                    alpha=0.5,
                    label='High LOF Score')
        
        # Customize the plot
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title('3D LOF Score Distribution', pad=20)
        
        # Add colorbar
        plt.colorbar(scatter3, ax=ax2, label='LOF Score', alpha=0.7)
        
        # Add grid
        ax2.grid(True, alpha=0.3)
        
       
        # Set initial view angle
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)
        
        print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")
    
        
        plt.show()


# Use physical CPU cores for heavy computation
PHYSICAL_CORES = psutil.cpu_count(logical=False)
# Use more threads for I/O bound operations
THREAD_COUNT = psutil.cpu_count(logical=True)

@dataclass
class LOFParams:
    """Dataclass to store LOF parameters"""
    n_neighbors: int
    contamination: float
    leaf_size: int
    p: int
    metric: str

    def to_dict(self) -> dict:
        """Convert dataclass to dictionary"""
        result = {
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'leaf_size': self.leaf_size,
            'p': self.p,
            'metric': self.metric
        }
        return result

def parallel_lof_scoring(params: LOFParams, X: np.ndarray) -> Tuple[float, LOFParams]:
    """Enhanced LOF scoring with improved outlier detection"""
    # 1. Add PCA preprocessing for better feature space representation
    pipeline = Pipeline([
        ('lof', LocalOutlierFactor(
            n_neighbors=params.n_neighbors,
            contamination=params.contamination,
            leaf_size=params.leaf_size,
            p=params.p,
            metric=params.metric,
            n_jobs=-1  # Parallel processing
        ))
    ])

    print(f"Processing params: {params}")

    # 2. Improve score calculation
    pipeline.fit(X)
    lof = pipeline.named_steps['lof']
    neg_factors = lof.negative_outlier_factor_

    # 3. Dynamic thresholding based on score distribution
    scores = -neg_factors  # Convert to positive scores for easier interpretation
    mad = np.median(np.abs(scores - np.median(scores)))
    thresh = np.median(scores) + (3 * mad)  # More robust than percentile

    # 4. Enhanced scoring metrics
    potential_outliers = scores > thresh

    # Improved separation with local density consideration
    k_distances = lof._distances_fit_X_
    local_density = np.mean(k_distances, axis=1)

    # Calculate relative density deviation
    density_ratio = local_density / (np.median(local_density) + 1e-8)  # Add epsilon to avoid division by zero

    # 5. Improved scoring components
    separation_score = np.abs(np.mean(scores[potential_outliers]) -
                             np.mean(scores[~potential_outliers])) / np.std(scores)

    density_score = np.mean(density_ratio[potential_outliers]) / np.mean(density_ratio)

    # 6. Consider local reachability
    reachability = np.max(k_distances, axis=1) / (np.min(k_distances, axis=1) + 1e-8)  # Add epsilon to avoid division by zero
    reach_score = np.mean(reachability[potential_outliers]) / (np.mean(reachability) + 1e-8)  # Add epsilon to avoid division by zero

    # 7. Weighted final score with emphasis on density
    final_score = (0.35 * separation_score +
                   0.40 * density_score +
                   0.25 * reach_score)

    return final_score, params

class LOFOutliersDetector:
    def __init__(self, contamination=0.1, n_neighbors=20):
        # Ensure contamination is within valid range
        self.contamination = min(max(contamination, 0.01), 0.5)
        
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('lof', LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=n_neighbors,
                n_jobs=psutil.cpu_count(logical=False)
            ))
        ])
        
        # Updated parameter distributions
        self.param_distributions = {
            'n_neighbors': randint(15, 50),
            'contamination': [0.01, 0.05, 0.1, 0.15, 0.2],
            'leaf_size': randint(20, 50),
            'p': [1, 2],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

    def _generate_param_combinations(self, n_iter: int) -> list:
        """Generate parameter combinations with enhanced constraints"""
        params_list = []
        for _ in range(n_iter):
            params = LOFParams(
                n_neighbors=np.random.randint(15, 50),
                contamination=np.random.choice([0.01, 0.05, 0.1, 0.15, 0.2]),
                leaf_size=np.random.randint(20, 50),
                p=np.random.choice([1, 2]),
                metric=np.random.choice(['euclidean', 'manhattan', 'minkowski'])
            )
            params_list.append(params)
        return params_list

    def _adaptive_contamination(self, scores: np.ndarray) -> float:
        """Adaptively determine contamination based on score distribution with bounds checking"""
        # Use Interquartile Range (IQR) method
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        threshold = q1 - 1.5 * iqr
        
        # Calculate adaptive contamination
        contamination = np.mean(scores < threshold)
        
        # Ensure contamination is within valid range (0, 0.5]
        contamination = min(max(contamination, 0.01), 0.5)
        
        return contamination

    def fit_predict_with_search(self, df: pd.DataFrame, n_iter: int = 30, plot_results: bool = True) -> pd.DataFrame:
        """Enhanced parameter search with adaptive contamination"""
        X = df.values
        params_list = self._generate_param_combinations(n_iter)
        
        # Parallel parameter search
        with ProcessPoolExecutor(max_workers=int(psutil.cpu_count(logical=False)/2.0)) as executor:
            results = list(executor.map(
                partial(parallel_lof_scoring, X=X),
                params_list
            ))
        
        # Find best parameters
        best_score, best_params = max(results, key=lambda x: x[0])
        
        # Pre-fit to get initial scores
        initial_lof = LocalOutlierFactor(contamination=0.1, n_neighbors=best_params.n_neighbors)
        initial_lof.fit(X)
        initial_scores = -initial_lof.negative_outlier_factor_
        
        # Calculate adaptive contamination with bounds checking
        adaptive_contamination = self._adaptive_contamination(initial_scores)
        
        # Configure pipeline with best parameters and adaptive contamination
        self.pipeline.named_steps['lof'].set_params(
            n_neighbors=best_params.n_neighbors,
            contamination=adaptive_contamination,
            leaf_size=best_params.leaf_size,
            p=best_params.p,
            metric=best_params.metric
        )
        
        # Final fit and predict
        outliers = self.pipeline.fit_predict(X)
        self.mask = outliers == -1
        lof_score = -self.pipeline.named_steps['lof'].negative_outlier_factor_
        # LOFParams to di
        if plot_results:
            self._print_results(df, best_params.to_dict(), best_score, lof_score)
            self._plot(df)
        
        return df[self.mask]
    
    def fit_predict(self, df: pd.DataFrame, plot_results: bool = True) -> pd.DataFrame:
        """Fit and predict outliers with LOF"""
        X = df.values
        outliers = self.pipeline.fit_predict(X)
        self.mask = outliers == -1
        
        if plot_results:
            self._print_results(df, self.pipeline.named_steps['lof'].get_params(), 0, self.pipeline.named_steps['lof'].negative_outlier_factor_)
            self._plot(df)
        
        return df[self.mask]

    def _plot(self, df):
        if df.shape[1] > 2:
            visualize_outliers_3d(
                df,
                self.pipeline.named_steps['lof'].negative_outlier_factor_,
                self.mask
            )
        else:
            visualize_outliers(
                df,
                self.pipeline.named_steps['lof'].negative_outlier_factor_,
                self.mask
            )

    def _print_results(self, df: pd.DataFrame, best_params: dict, best_score: float, anomaly_scores: np.ndarray):
        """Print formatted results of LOF detection."""
        print("\n" + "="*50)
        print("Local Outlier Factor (LOF) Detection Results")
        print("="*50)
        
        # Model Configuration
        print("\nBest Parameters:")
        print(f"  n_neighbors:  {best_params['n_neighbors']}")
        print(f"  leaf_size:    {best_params['leaf_size']}")
        print(f"  metric:       {best_params['metric']}")
        print(f"  p:            {best_params['p']}")
        print(f"  Score:        {best_score:.4f}")
        
        # Calculate outlier mask using adaptive contamination
        contamination = self._adaptive_contamination(anomaly_scores)
        threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
        mask = anomaly_scores > threshold
        
        # Detection Statistics
        n_outliers = mask.sum()
        outlier_ratio = n_outliers/len(df)
        print("\nDetection Statistics:")
        print(f"  Total points:      {len(df):,}")
        print(f"  Outliers found:    {n_outliers:,} ({outlier_ratio:.1%})")
        print(f"  Inliers retained:  {len(df)-n_outliers:,} ({1-outlier_ratio:.1%})")
        print(f"  Contamination:     {contamination:.1%}")
        
        # Score Distribution
        print("\nLOF Score Distribution:")
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


# Optimize visualization functions
def visualize_outliers(df: pd.DataFrame, lof_scores: np.ndarray, mask: np.ndarray,
                      fig_size: Tuple[int, int] = (15, 7)) -> None:
    """Optimized 2D visualization"""
    with plt.style.context('seaborn-v0_8-whitegrid'):  # Use context manager for style
        fig = plt.figure(figsize=fig_size,  constrained_layout=True)
        gs = plt.GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Perform PCA once and reuse results
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df.values)
        
        # Plot 1: PCA visualization
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(X_pca[~mask, 0], X_pca[~mask, 1], 
                   c='royalblue', label='Inliers', alpha=0.6, s=50)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c='crimson', label='Outliers', alpha=0.8, s=100)
        
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_title('PCA Projection with Outliers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: LOF scores
        ax2 = fig.add_subplot(gs[1])
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=lof_scores, cmap='RdYlBu_r',
                            s=100, alpha=0.6)
        
        plt.colorbar(scatter, ax=ax2, label='LOF Score')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('LOF Score Distribution')
        ax2.grid(True, alpha=0.3)
        print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")

        plt.show()



    
__all__ = ['LOFOutliersDetector']


if __name__ == "__main__":
    # Test the LOFOutliersDetector class
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=1, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    detector = LOFOutliersDetector()
    # print(outliers)
    outliers = detector.fit_predict_with_search(df)
    detector.fit_predict(df)
    # print(outliers)