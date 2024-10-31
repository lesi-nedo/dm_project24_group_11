import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import psutil

from matplotlib.legend_handler import HandlerPathCollection

from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from scipy.stats import randint


real_cpu_count = psutil.cpu_count(logical=False)    # Number of physical cores

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

def visualize_outliers(df: pd.DataFrame, lof_scores: np.ndarray, mask: np.ndarray) -> None:
    """
    Create enhanced visualizations for outlier detection results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to detect outliers
    lof_scores : numpy.ndarray
        LOF scores from the model
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
    X = df.values
    X_pca = pca.fit_transform(X)
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
    ax1.set_title('PCA Projection of Data\nwith Outlier Detection', pad=20)
    ax1.legend()
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LOF scores visualization
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate normalized radius for circles
    radius = (lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min())
    
    # Create enhanced LOF visualization
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
               color="darkblue", s=30, alpha=0.6, label="Data points")
    
    # Plot circles with radius proportional to the outlier scores
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                         s=1000 * radius,
                         edgecolors="crimson",
                         facecolors="none",
                         alpha=0.5,
                         label="Outlier scores")
    
    # Customize the LOF plot
    ax2.set_title('Local Outlier Factor (LOF) Scores\nCircle Size ∝ Anomaly Score', pad=20)
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.grid(True, alpha=0.3)
    
    # Add legend with custom handler
    def update_legend_marker_size(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([100])
    
    ax2.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
    
    # Add colorbar to show LOF score distribution
    norm = plt.Normalize(lof_scores.min(), lof_scores.max())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
    plt.colorbar(sm, ax=ax2, label='LOF Score', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Additional statistical summary
    print("\nOutlier Detection Summary:")
    print(f"Total points: {len(df)}")
    print(f"Outliers detected: {mask.sum()} ({mask.sum()/len(df):.1%})")
    print(f"LOF Score Range: {lof_scores.min():.2f} to {lof_scores.max():.2f}")


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
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 8))
    
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
    
    # Add rotation animation
    def rotate(angle):
        ax1.view_init(azim=angle)
        ax2.view_init(azim=angle)
        plt.draw()
    
    # Function to update the view angle
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
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Set initial view angle
    ax1.view_init(elev=20, azim=45)
    ax2.view_init(elev=20, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Print summary statistics
    print("\n3D Outlier Detection Summary:")
    print(f"Total points: {len(df)}")
    print(f"Outliers detected: {mask.sum()} ({mask.sum()/len(df):.1%})")
    print(f"LOF Score Range: {lof_scores.min():.2f} to {lof_scores.max():.2f}")
    print(f"Variance explained by 3 PCs: {sum(variance_ratio):.1%}")
    print("\nInteractive Controls:")
    print("- Use arrow keys to rotate and tilt the plots")
    print("- Close the plot window to exit")
    
    plt.show()


def lof_scoring(estimator, X):
    """
    Advanced scoring function for Local Outlier Factor optimization.
    
    Combines multiple metrics to evaluate LOF performance:
    1. Separation between inlier and outlier scores
    2. Local density consistency
    3. Distance-based clustering quality
    
    Parameters:
    -----------
    estimator : Pipeline
        Fitted pipeline containing LOF estimator
    X : array-like
        Input data to evaluate
    
    Returns:
    --------
    float
        Composite score where higher values indicate better outlier detection
    """
    # Get LOF estimator from pipeline
    lof = estimator.named_steps['lof']
    
    # Fit LOF and get scores
    lof.fit(X)
    neg_factors = lof.negative_outlier_factor_
    
    # Calculate score components
    
    # 1. Score separation
    thresh = np.percentile(neg_factors, 90)  # Top 10% threshold
    potential_outliers = neg_factors < thresh
    separation_score = np.abs(np.mean(neg_factors[potential_outliers]) - 
                            np.mean(neg_factors[~potential_outliers]))
    
    # 2. Local density consistency
    density_var = np.std(neg_factors) / (np.max(neg_factors) - np.min(neg_factors))
    density_score = 1 / (1 + density_var)  # Normalize to [0,1]
    
    # 3. Distance-based quality
    distances = lof._distances_fit_X_
    avg_neighbor_dist = np.mean(distances, axis=1)
    distance_score = 1 / (1 + np.std(avg_neighbor_dist))
    
    # Combine scores with weights
    final_score = (0.5 * separation_score + 
                  0.3 * density_score +
                  0.2 * distance_score)
    
    return final_score

class LOFOutliersDetector:

    def __init__(self, contamination='auto', n_neighbors=10) -> None:
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lof', LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors))
        ])

        self.param_distributions = {
            'lof__n_neighbors': randint(5, 50),
            'lof__contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
            'lof__leaf_size': randint(20, 50),
            'lof__p': [1, 2, 3]
        }
    

    def fit_predict_with_search(self, df: pd.DataFrame, plot_results=True, n_iter: int=20, cv: int=5, verbose: int=1) -> pd.DataFrame:
        """
            Fit and predict outliers using Local Outlier Factor with RandomizedSearchCV

            Parameters:
            -----------
            df : pandas.DataFrame
                DataFrame containing features to detect outliers
            
            
            n_iter : int
                Number of iterations to run RandomizedSearchCV
            
            cv : int
                Number of cross-validation folds

            verbose : bool
                Verbosity level
            
            Returns:
            --------
            pandas.DataFrame
                DataFrame containing outliers
            
        """
        X = df.values
        warnings.filterwarnings("ignore")
        search = RandomizedSearchCV(
            self.pipeline, 
            self.param_distributions, 
            n_iter=n_iter, 
            cv=cv, 
            n_jobs=int(real_cpu_count * 0.3),
            verbose=verbose,
            scoring=make_scorer(lof_scoring)
        )
        search.fit(X)
        best_estimator = search.best_estimator_
        outliers = best_estimator.fit_predict(X)
        self.mask = outliers == -1

        if plot_results:
            self._visualize_outliers(df)

        warnings.filterwarnings("default")

        return df[self.mask]
    
    def fit_predict(self, df: pd.DataFrame, plot_results=True) -> pd.DataFrame:
        """
            Fit and predict outliers using Local Outlier Factor

            Parameters:
            -----------
            df : pandas.DataFrame
                DataFrame containing features to detect outliers
            
            Returns:
            --------
            pandas.DataFrame
                DataFrame containing outliers
            
        """
        X = df.values
        outliers = self.pipeline.fit_predict(X)
        self.mask = outliers == -1

        if plot_results and df.shape[1] > 2:
            visualize_outliers_3d(df, self.pipeline.named_steps['lof'].negative_outlier_factor_, self.mask)
        elif plot_results:
            visualize_outliers(df, self.pipeline.named_steps['lof'].negative_outlier_factor_, self.mask)

        return df[self.mask]
    


    
__all__ = ['LOFOutliersDetector']


if __name__ == "__main__":
    # Test the LOFOutliersDetector class
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=1, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    detector = LOFOutliersDetector()
    # print(outliers)
    outliers = detector.fit_predict(df)
    # print(outliers)