import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


from sklearn.decomposition import PCA

from typing import Dict

    

def calculate_z_score(data_tuple):
    """
    Wrapper function for z-score calculation
    """
    outliers_data, mean_data, std_data = map(np.asarray, data_tuple)


    return np.divide(
        np.subtract(outliers_data, mean_data),
        std_data,
        out=np.zeros_like(outliers_data),
        where=std_data!=0
    )

def plot_outlier_scores(z_score_lof: np.ndarray, 
                       z_score_iso_for: np.ndarray, 
                       z_score_oc_svm: np.ndarray,
                       figsize: tuple = (15, 10)) -> None:
    """Creates visualization plots for outlier detection z-scores."""
    
    def _truncate_pair(arr1, arr2):
        """Truncate two arrays to the shorter length"""
        min_len = min(len(arr1), len(arr2))
        return arr1[:min_len], arr2[:min_len]

    # Ensure arrays are 1-dimensional
    z_score_lof = np.ravel(z_score_lof)
    z_score_iso_for = np.ravel(z_score_iso_for)
    z_score_oc_svm = np.ravel(z_score_oc_svm)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize)
    
    # Histograms with full data
    sns.histplot(data=z_score_lof, ax=ax1, bins='auto', color='skyblue')
    ax1.set_title('LOF Z-Scores Distribution')
    ax1.set_xlabel('Z-Score')
    ax1.set_xlim(-5, 5)
    
    sns.histplot(data=z_score_iso_for, ax=ax2, bins='auto', color='lightgreen')
    ax2.set_title('Isolation Forest Z-Scores Distribution')
    ax2.set_xlabel('Z-Score')
    ax2.set_xlim(-5, 5)
    
    sns.histplot(data=z_score_oc_svm, ax=ax3, bins='auto', color='salmon')
    ax3.set_title('One-Class SVM Z-Scores Distribution')
    ax3.set_xlabel('Z-Score')
    ax3.set_xlim(-5, 5)
    
    # Scatter plots with truncated pairs
    lof_iso, iso_for_trunc = _truncate_pair(z_score_lof, z_score_iso_for)
    sns.scatterplot(x=lof_iso, y=iso_for_trunc, ax=ax4, alpha=0.6)
    ax4.set_title('LOF vs Isolation Forest')
    ax4.set_xlabel('LOF Z-Score')
    ax4.set_ylabel('Isolation Forest Z-Score')
    
    lof_svm, oc_svm_trunc1 = _truncate_pair(z_score_lof, z_score_oc_svm)
    sns.scatterplot(x=lof_svm, y=oc_svm_trunc1, ax=ax5, alpha=0.6)
    ax5.set_title('LOF vs One-Class SVM')
    ax5.set_xlabel('LOF Z-Score')
    ax5.set_ylabel('One-Class SVM Z-Score')
    
    iso_svm, oc_svm_trunc2 = _truncate_pair(z_score_iso_for, z_score_oc_svm)
    sns.scatterplot(x=iso_svm, y=oc_svm_trunc2, ax=ax6, alpha=0.6)
    ax6.set_title('Isolation Forest vs One-Class SVM')
    ax6.set_xlabel('Isolation Forest Z-Score')
    ax6.set_ylabel('One-Class SVM Z-Score')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_z_score_distributions(z_scores_dict: Dict[str, np.ndarray], 
                             figsize: tuple = (15, 5),
                             title: str = "Z-Score Distributions by Algorithm") -> None:
    """
    Creates visualization plots for multiple algorithm z-score distributions.
    
    Args:
        z_scores_dict: Dictionary mapping algorithm names to their z-scores
        figsize: Figure size tuple (width, height)
        title: Main title for the plot
    """
    # Input validation
    if not z_scores_dict:
        raise ValueError("z_scores_dict cannot be empty")
    
    # Ensure all arrays are 1-dimensional
    for name, scores in z_scores_dict.items():
        z_scores_dict[name] = np.ravel(scores)
        if z_scores_dict[name].ndim != 1:
            raise ValueError(f"Array for {name} must be 1-dimensional after flattening")

    # Create subplot grid
    fig, axes = plt.subplots(1, len(z_scores_dict), figsize=figsize)
    
    # Handle single algorithm case
    if len(z_scores_dict) == 1:
        axes = [axes]

    # Color mapping
    colors = {'LOF': 'skyblue', 
              'Isolation Forest': 'lightgreen', 
              'One-Class SVM': 'salmon'}
    
    # Create distributions
    for (name, scores), ax in zip(z_scores_dict.items(), axes):
        color = colors.get(name, 'gray')
        sns.boxplot(data=scores, ax=ax, color=color)
        ax.set_title(f'{name} Z-Scores')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_kdes(df_dicts: dict[str, pd.DataFrame], figsize: tuple = (18, 6), title="KDE Plot of the Data"):
    fig, axes = plt.subplots(1, len(df_dicts), figsize=figsize)
    
    for (name, dfs), ax in zip(df_dicts.items(), axes):
        outliers, inliers = dfs
        
        # Adjusting KDE bandwidth for smoother plots
        sns.kdeplot(data=outliers, ax=ax, label='Outliers', color='red', bw_adjust=0.5)
        sns.kdeplot(data=inliers, ax=ax, label='Inliers', color='blue', bw_adjust=0.5)
        
        ax.set_title(f'{name} KDE Plot')
        # ax.set_yscale('log')  # Log scale to handle peaks
        ax.tick_params(axis='x', rotation=45)
        
        # Adding grid and adjusting limits
        ax.grid(True)
        ax.set_xlim(-50000, 50000)  # Example x-limit for better visibility
        
        # Handling legend
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 2:
            ax.legend(handles=handles[:2], labels=['Outliers', 'Inliers'], loc='upper right')  # Adjust position if needed
        else:
            ax.legend(loc='upper right')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to prevent title overlap
    plt.show()
    plt.close()






def visualize_outliers_3d(data, lof_set, iso_set, oc_svm_set):
    """
    Reduce dimensionality of data using PCA and create a 3D visualization of outliers
    detected by different algorithms.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original dataset containing all features
    lof_set : set
        Set of indices for outliers detected by LOF
    iso_set : set
        Set of indices for outliers detected by Isolation Forest
    oc_svm_set : set
        Set of indices for outliers detected by One-Class SVM
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive 3D scatter plot
    """
    # Perform PCA
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    
    # Create a DataFrame with PCA components
    pca_df = pd.DataFrame(
        data_pca, 
        columns=['PC1', 'PC2', 'PC3'],
        index=data.index
    )
    
    # Calculate intersections and unique points
    all_outliers = lof_set | iso_set | oc_svm_set
    lof_iso = lof_set & iso_set
    lof_oc_svm = lof_set & oc_svm_set
    iso_oc_svm = iso_set & oc_svm_set
    all_common = lof_set & iso_set & oc_svm_set
    
    # Create category labels for each point
    categories = []
    for idx in data.index:
        if idx in all_common:
            categories.append('Detected by All')
        elif idx in lof_iso:
            categories.append('LOF & IsoForest')
        elif idx in lof_oc_svm:
            categories.append('LOF & OC-SVM')
        elif idx in iso_oc_svm:
            categories.append('IsoForest & OC-SVM')
        elif idx in lof_set:
            categories.append('LOF only')
        elif idx in iso_set:
            categories.append('IsoForest only')
        elif idx in oc_svm_set:
            categories.append('OC-SVM only')
        else:
            categories.append('Normal')
    

    pca_df['Category'] = categories
    
    # Create color scheme
    color_discrete_map = {
        'Detected by All': '#FF0000',      # Red
        'LOF & IsoForest': '#FFA500',      # Orange
        'LOF & OC-SVM': '#FFFF00',         # Yellow
        'IsoForest & OC-SVM': '#800080',   # Purple
        'LOF only': '#0000FF',             # Blue
        'IsoForest only': '#008000',       # Green
        'OC-SVM only': '#000000',          # Black
        'Normal': '#808080'                # Gray
    }
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Category',
        color_discrete_map=color_discrete_map,
        title='3D Visualization of Outliers (PCA)',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)'
        }
    )
    
    # Update layout for better visualization
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        legend_title_text='Detection Category',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )
    
    return fig, pca_df

def print_summary_statistics(lof_set, iso_set, oc_svm_set):
    """
    Print summary statistics about the outliers detected by each algorithm
    and their intersections.
    """
    all_common = lof_set & iso_set & oc_svm_set
    lof_iso = lof_set & iso_set
    lof_oc_svm = lof_set & oc_svm_set
    iso_oc_svm = iso_set & oc_svm_set
    
    print("Summary Statistics:")
    print(f"Total outliers detected by LOF: {len(lof_set)}")
    print(f"Total outliers detected by Isolation Forest: {len(iso_set)}")
    print(f"Total outliers detected by OC-SVM: {len(oc_svm_set)}")
    print("\nIntersections:")
    print(f"Detected by all three methods: {len(all_common)}")
    print(f"Common between LOF and IsoForest: {len(lof_iso)}")
    print(f"Common between LOF and OC-SVM: {len(lof_oc_svm)}")
    print(f"Common between IsoForest and OC-SVM: {len(iso_oc_svm)}")