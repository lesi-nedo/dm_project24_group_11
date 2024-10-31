import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil



from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
from typing import Tuple, Optional
from scipy.stats import randint, uniform
import warnings


real_cpu_count = psutil.cpu_count(logical=False)


def visualize_isolation_forest(df: pd.DataFrame, anomaly_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced visualizations for Isolation Forest outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    anomaly_scores : numpy.ndarray
        Anomaly scores from the model (-1 for outliers, 1 for inliers)
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
    ax1.set_title('PCA Projection of Data\nwith Isolation Forest Detection', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly Score visualization
    ax2 = fig.add_subplot(gs[1])
    
    # Normalize scores for visualization
    normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Create scatter plot with size proportional to anomaly score
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=anomaly_scores,
                         cmap='RdYlBu_r',
                         s=1000 * normalized_scores,
                         alpha=0.6,
                         label='Anomaly scores')
    
    # Customize the plot
    ax2.set_title('Isolation Forest Anomaly Scores\nMarker Size ∝ Anomaly Score', pad=20)
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
    plt.colorbar(sm, ax=ax2, label='Anomaly Score', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print("\nIsolation Forest Detection Summary:")
    print(f"Total points: {len(df)}")
    print(f"Outliers detected: {mask.sum()} ({mask.sum()/len(df):.1%})")
    print(f"Anomaly Score Range: {anomaly_scores.min():.2f} to {anomaly_scores.max():.2f}")

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
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 8))
    
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
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n3D Isolation Forest Detection Summary:")
    print(f"Total points: {len(df)}")
    print(f"Outliers detected: {mask.sum()} ({mask.sum()/len(df):.1%})")
    print(f"Anomaly Score Range: {anomaly_scores.min():.2f} to {anomaly_scores.max():.2f}")
    print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")
    print("\nInteractive Controls:")
    print("- Use arrow keys to rotate and tilt the plots")
    print("- Close the plot window to exit")
    
    plt.show()


def isolation_forest_scoring(estimator, X):
    """
    Custom scoring function for IsolationForest.
    Combines average path length deviation and clustering metrics.
    
    Parameters:
    -----------
    estimator : Pipeline
        The pipeline containing IsolationForest
    X : array-like
        The input samples
    
    Returns:
    --------
    float
        Score value (higher is better)
    """
    # Get the IsolationForest from pipeline
    iforest = estimator.named_steps['iforest']
    
    # Get decision function scores
    scores = iforest.decision_function(X)
    
    # Calculate metrics
    avg_path_length = np.mean([e.decision_path(X)[1].data.mean() 
                              for e in iforest.estimators_])
    score_std = np.std(scores)  # Higher std suggests better separation
    
    # Combine metrics (normalized path length and score standard deviation)
    # We want shorter paths (more anomalous) and higher standard deviation
    normalized_path = 1 / (1 + avg_path_length)
    combined_score = (normalized_path + score_std) / 2
    
    return combined_score

class IsolationForestDetector:
    """
    Enhanced Isolation Forest-based outlier detector with visualization capabilities
    and hyperparameter search.
    """
    
    def __init__(
        self,
        contamination: float = 'auto',
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the Isolation Forest detector with enhanced parameters.
        
        Parameters:
        -----------
        contamination : float or 'auto', default='auto'
            Expected proportion of outliers in the dataset
        n_estimators : int, default=100
            Number of base estimators in the ensemble
        max_samples : int or str, default='auto'
            Number of samples to draw to train each base estimator
        max_features : float, default=1.0
            Number of features to draw from X to train each base estimator
        bootstrap : bool, default=False
            Whether to use bootstrap when drawing samples
        random_state : int or None, default=None
            Random state for reproducibility
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('iforest', IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
        # Define parameter distributions for random search
        self.param_distributions = {
            'iforest__n_estimators': randint(50, 200),
            'iforest__max_samples': uniform(0.1, 0.9),  # proportion of samples
            'iforest__max_features': uniform(0.1, 0.9),  # proportion of features
            'iforest__contamination': uniform(0.01, 0.3),
            'iforest__bootstrap': [True, False]
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
        Fit and predict outliers using Isolation Forest with RandomizedSearchCV
        
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
            n_jobs=int(real_cpu_count *0.2),
            verbose=verbose,
            scoring=make_scorer(isolation_forest_scoring)
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
        
        # Get anomaly scores
        anomaly_scores = -best_estimator.named_steps['iforest'].decision_function(X)
        
        # Update pipeline with best estimator
        self.pipeline = best_estimator
        
        # Visualize results if requested
        if plot_results and df.shape[1] > 2:
            visualize_isolation_forest_3d(df, anomaly_scores, self.mask)
        elif plot_results:
            visualize_isolation_forest(df, anomaly_scores, self.mask)
        
        warnings.filterwarnings("default")
        
        return df[self.mask], anomaly_scores
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        plot_results: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the model and predict outliers using Isolation Forest.
        
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
        
        # Get anomaly scores (negative of decision function for consistency with LOF)
        anomaly_scores = -self.pipeline.named_steps['iforest'].decision_function(X)
        
        # Visualize results if requested
        if plot_results and df.shape[1] > 2:
            visualize_isolation_forest_3d(df, anomaly_scores, self.mask)
        elif plot_results:
            visualize_isolation_forest(df, anomaly_scores, self.mask)
        
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
    

__all__ = ['IsolationForestDetector']