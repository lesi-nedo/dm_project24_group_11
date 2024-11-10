import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import psutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple, Optional
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from scipy.stats import randint, uniform
from scipy import stats
import warnings


# Use physical CPU cores for computation
PHYSICAL_CORES = psutil.cpu_count(logical=False)
# Use more threads for I/O bound operations
THREAD_COUNT = psutil.cpu_count(logical=True)


def visualize_isolation_forest_3d(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced 3D visualizations for Isolation Forest outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    anomaly_scores : numpy.ndarray
        Anomaly scores from the model
    mask : numpy.ndarray
        Boolean mask indicating outliers
    """
    with plt.style.context('seaborn-v0_8-whitegrid'):

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8), constrained_layout=True)
        
        # Separate outliers and inliers
        outliers = df[mask]
        inliers = df[~mask]
        
        # Apply PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(df.values)
        X_outliers = pca.transform(outliers.values)
        X_inliers = pca.transform(inliers.values)
        
        # Plot 1: 3D PCA visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot inliers and outliers
        ax1.scatter(X_inliers[:, 0], X_inliers[:, 1], X_inliers[:, 2],
                    c='royalblue', label='Inliers', alpha=0.6, s=50)
        ax1.scatter(X_outliers[:, 0], X_outliers[:, 1], X_outliers[:, 2],
                    c='crimson', label='Outliers', alpha=0.8, s=100)
        
        # Customize the plot
        variance_ratio = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} var)')
        ax1.set_zlabel(f'PC3 ({variance_ratio[2]:.1%} var)')
        ax1.set_title('3D PCA Projection with Isolation Forest', pad=20)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 3D Anomaly Score visualization
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Create scatter plot with color based on anomaly scores
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                            c=anomaly_scores,
                            cmap='RdYlBu_r',
                            s=100,
                            alpha=0.6)
        
        # Add spheres for high anomaly scores
        high_anomaly_mask = anomaly_scores < np.percentile(anomaly_scores, 10)  # Top 10% most anomalous
        if np.any(high_anomaly_mask):
            ax2.scatter(X_pca[high_anomaly_mask, 0],
                    X_pca[high_anomaly_mask, 1],
                    X_pca[high_anomaly_mask, 2],
                    s=200,
                    facecolors='none',
                    edgecolors='red',
                    alpha=0.5,
                    label='High Anomaly Score')
        
        # Customize the plot
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title('3D Anomaly Score Distribution', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score', alpha=0.7)
        
        # Add rotation controls
        def rotate(angle):
            ax1.view_init(azim=angle)
            ax2.view_init(azim=angle)
            plt.draw()
        
        def on_key_press(event):
            if event.key == 'left':
                rotate(ax1.azim - 5)
            elif event.key == 'right':
                rotate(ax1.azim + 5)
            elif event.key == 'up':
                ax1.view_init(elev=ax1.elev + 5)
                ax2.view_init(elev=ax2.elev + 5)
                plt.draw()
            elif event.key == 'down':
                ax1.view_init(elev=ax1.elev - 5)
                ax2.view_init(elev=ax2.elev - 5)
                plt.draw()
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Set initial view angle
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)
        
        print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")
        print("\nInteractive Controls:")
        print("- Use arrow keys to rotate and tilt the plots")
        print("- Close the plot window to exit")
        
        plt.show()



@dataclass
class IForestParams:
    """Dataclass to store Isolation Forest parameters"""
    n_estimators: int
    max_samples: float
    max_features: float
    contamination: float
    bootstrap: bool

def parallel_iforest_scoring(params: IForestParams, X: np.ndarray) -> Tuple[float, IForestParams]:
    """Enhanced parallel Isolation Forest scoring with extreme value control"""
    pipeline = Pipeline([
        ('iforest', IsolationForest(
            n_estimators=params.n_estimators,
            max_samples=params.max_samples,
            max_features=params.max_features,
            contamination=params.contamination,
            bootstrap=params.bootstrap
        ))
    ])
    
    pipeline.fit(X)
    scores = pipeline.named_steps['iforest'].decision_function(X) + 0.5
    # Core metrics
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_skew = np.abs(stats.skew(scores))
    score_kurtosis = stats.kurtosis(scores)
    
    # Extreme value analysis - now used
    extreme_threshold = score_mean - (3 * score_std)
    extreme_ratio = np.sum(scores < extreme_threshold) / len(scores)
    extreme_penalty = np.exp(extreme_ratio ) if extreme_ratio > 0.01 else 1.0
    
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
    
    # Strong contamination penalty
    contamination_penalty = np.exp(params.contamination * 15)
    combined_score = combined_score / contamination_penalty
    
    return 1 / (1 + np.exp(-combined_score)), params


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 'auto',
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None
    ) -> None:
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('iforest', IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=PHYSICAL_CORES
            ))
        ])
        
        self.param_distributions = {
            'n_estimators': randint(100, 300),
            'max_samples': uniform(0.1, 0.4),
            'max_features': uniform(0.1, 0.9),
            'contamination': uniform(0.001, 0.1),
            'bootstrap': [True]
        }

    def _generate_param_combinations(self, n_iter: int) -> list:
        """Generate parameter combinations for parallel processing"""
        params_list = []
        for _ in range(n_iter):
            params = IForestParams(
                n_estimators=self.param_distributions['n_estimators'].rvs(),
                max_samples=self.param_distributions['max_samples'].rvs(),
                max_features=self.param_distributions['max_features'].rvs(),
                contamination=self.param_distributions['contamination'].rvs(),
                bootstrap=np.random.choice(self.param_distributions['bootstrap'])
            )
            params_list.append(params)
        return params_list

    def _plot(self, df, anomaly_scores):
        if df.shape[1] > 2:
            visualize_isolation_forest_3d(
                df,
                anomaly_scores,
                self.mask
            )
        else:
            visualize_isolation_forest(
                df,
                anomaly_scores,
                self.mask
            )
    def _print_results(self, df: pd.DataFrame, best_params: dict, best_score: float, anomaly_scores: np.ndarray):
        """Print formatted results of Isolation Forest detection."""
        print("\n" + "="*50)
        print("Isolation Forest Detection Results")
        print("="*50)
        
        # Model Configuration
        print("\nBest Parameters:")
        print(f"  n_estimators:      {best_params['n_estimators']}")
        print(f"  max_samples:       {best_params['max_samples']}")
        print(f"  contamination:     {best_params['contamination']}")
        print(f"  max_features:      {best_params['max_features']}")
        print(f"  bootstrap:         {best_params['bootstrap']}")
        print(f"  Score:            {best_score:.4f}")
        
        # Handle 'auto' contamination by using a default value of 0.1
        contamination = 0.1 if best_params['contamination'] == 'auto' else float(best_params['contamination'])
        
        # Calculate outlier mask using provided contamination
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
        print("\nIsolation Forest Score Distribution:")
        print(f"  Min score:     {anomaly_scores.min():.2f}")
        print(f"  Max score:     {anomaly_scores.max():.2f}")
        print(f"  Mean score:    {np.mean(anomaly_scores):.2f}")
        print(f"  Median score:  {np.median(anomaly_scores):.2f}")
        
        # Percentiles 
        percentiles = np.percentile(anomaly_scores, [25, 75, 90, 95, 99])
        print("\nScore Percentiles:")
        print(f"  25th: {percentiles[0]:.2f}")
        print(f"  75th: {percentiles[1]:.2f}")

    def fit_predict_with_search(
        self,
        df: pd.DataFrame,
        plot_results: bool = True,
        n_iter: int = 20,
        verbose: int = 1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Optimized parallel parameter search"""
        X = df.values
        params_list = self._generate_param_combinations(n_iter)
        
        with ProcessPoolExecutor(max_workers=PHYSICAL_CORES) as executor:
            results = list(executor.map(
                partial(parallel_iforest_scoring, X=X),
                params_list
            ))
        
        best_score, best_params = max(results, key=lambda x: x[0])
        
        self.pipeline.named_steps['iforest'].set_params(
            n_estimators=best_params.n_estimators,
            max_samples=best_params.max_samples,
            max_features=best_params.max_features,
            contamination=best_params.contamination,
            bootstrap=best_params.bootstrap
        )
        
        predictions = self.pipeline.fit_predict(X)
        self.mask = predictions == -1
        anomaly_scores = -self.pipeline.named_steps['iforest'].decision_function(X)
        
        if plot_results:
            self._print_results(df, best_params.__dict__, best_score, anomaly_scores)
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
        anomaly_scores = -self.pipeline.named_steps['iforest'].decision_function(X)
        
        if plot_results:
            self._print_results(df, self.pipeline.named_steps['iforest'].get_params(), 0.0, anomaly_scores)
            self._plot(df, anomaly_scores)
        
        return df[self.mask], anomaly_scores

# Optimize visualization functions (similar to previous implementation)
def visualize_isolation_forest(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray,
                             fig_size: Tuple[int, int] = (15, 7)) -> None:
    """Optimized 2D visualization"""
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=fig_size, constrained_layout=True)

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
        ax1.set_title('PCA Projection with Isolation Forest')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        ax2 = fig.add_subplot(gs[1])
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=anomaly_scores, cmap='RdYlBu_r',
                            s=100, alpha=0.6)
        
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Anomaly Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.show()

        print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")



__all__ = ['IsolationForestDetector']


if __name__ == "__main__":
    # Test the LOFOutliersDetector class
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=1, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    detector = IsolationForestDetector()
    # print(outliers)
    outliers = detector.fit_predict_with_search(df)
    # detector.fit_predict(df)
    # print(outliers)