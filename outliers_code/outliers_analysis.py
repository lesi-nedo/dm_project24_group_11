import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os


from sklearn.decomposition import PCA
from collections import defaultdict

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
    
    return fig

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
    return fig

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
        'OC-SVM only': '#FFC0CB',          # Pink
        'Normal': '#808080'                # Gray
    }
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each category with different colors
        for category, color in color_discrete_map.items():
            mask = pca_df['Category'] == category
            ax.scatter(
                pca_df[mask]['PC1'],
                pca_df[mask]['PC2'],
                pca_df[mask]['PC3'],
                c=color,
                label=category,
                s=50,  # marker size
                alpha=0.6  # transparency
            )

        # Set labels and title
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
        ax.set_title('3D Visualization of Outliers (PCA)')

        # Add legend
        ax.legend(title='Detection Category')

        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=45)

    del pca_df





def analyze_outlier_types(data, outlier_indices, min_races_threshold=1):
    """
    Analyze whether outliers are primarily associated with specific cyclists or races.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original dataset
    outlier_indices : set
        Set of indices marking outlier points
    min_races_threshold : int
        Minimum number of races to consider a pattern significant
        
    Returns:
    --------
    dict : Dictionary containing cyclist and race outlier classifications
    """
    # Count occurrences for cyclists and races
    cyclist_counts = defaultdict(int)
    race_counts = defaultdict(int)
    
    # Count total appearances for each cyclist and race
    cyclist_totals = data['cyclist'].value_counts()
    race_totals = data['_url_race'].value_counts()
    
    # Count outlier occurrences
    for idx in outlier_indices:
        cyclist = data.loc[idx, 'cyclist']
        race = data.loc[idx, '_url_race']
        cyclist_counts[cyclist] += 1
        race_counts[race] += 1
    
    # Calculate outlier ratios
    cyclist_ratios = {
        cyclist: count / cyclist_totals[cyclist]
        for cyclist, count in cyclist_counts.items()
        if cyclist_totals[cyclist] >= min_races_threshold
    }
    
    race_ratios = {
        race: count / race_totals[race]
        for race, count in race_counts.items()
        if race_totals[race] >= min_races_threshold
    }
    
    # Classify outliers
    cyclist_outliers = set()
    race_outliers = set()
    
    # If a cyclist/race has more than 50% of their entries as outliers,
    # consider them systematic outliers
    threshold = 0.2
    
    for idx in outlier_indices:
        cyclist = data.loc[idx, 'cyclist']
        race = data.loc[idx, '_url_race']
        
        if cyclist in cyclist_ratios and cyclist_ratios[cyclist] > threshold:
            cyclist_outliers.add(idx)
        elif race in race_ratios and race_ratios[race] > threshold:
            race_outliers.add(idx)
    
    print(f"Total cyclist outliers: {len(cyclist_outliers)}")
    print(f"Total race outliers: {len(race_outliers)}")
            
    return {
        'cyclist_outliers': cyclist_outliers,
        'race_outliers': race_outliers,
        'cyclist_ratios': cyclist_ratios,
        'race_ratios': race_ratios
    }

def visualize_outlier_types(data, all_outliers_indx, min_races_threshold=5, sample_size=8000, random_state=42):
    """
    Create separate visualizations for cyclist and race outliers.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original dataset
    lof_set, iso_set, oc_svm_set : set
        Sets of indices for outliers detected by different methods
    min_races_threshold : int
        Minimum number of races to consider a pattern significant
    
    Returns:
    --------
    tuple : (cyclist_fig, race_fig, summary_dict)
    """
    # Combine all outliers

    len_outliers = len(all_outliers_indx)
    len_data = len(data)
    
    # Analyze outlier types using full dataset
    outlier_analysis = analyze_outlier_types(data, all_outliers_indx, min_races_threshold)
    cyclist_outliers = outlier_analysis['cyclist_outliers']
    race_outliers = outlier_analysis['race_outliers']
    
    # Sample indices while preserving outlier proportions
    normal_indices = set(range(len_data)) - all_outliers_indx
    n_outliers = min(len_outliers, int(sample_size * len_outliers / len_data))
    n_normal = sample_size - n_outliers
    
    np.random.seed(random_state)
    sampled_normal = set(np.random.choice(list(normal_indices), n_normal, replace=False))
    sampled_indices = list(sampled_normal.union(all_outliers_indx))
    
    # Sample data
    sampled_data = data.iloc[sampled_indices].copy()
    data = sampled_data

    # Select numerical features for PCA
    numerical_features = [
        'birth_year', 'weight', 'height', 'points', 'uci_points',
        'length', 'climb_total', 'profile', 'startlist_quality',
        'cyclist_age', 'delta'
    ]
    
    # Prepare data for PCA
    X = data[numerical_features]
    
    # Perform PCA
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(X)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        data_pca,
        columns=['PC1', 'PC2', 'PC3'],
        index=data.index
    )
    
    # Add categories for cyclist visualization
    pca_df['Cyclist_Category'] = 'Normal'
    pca_df.loc[list(cyclist_outliers), 'Cyclist_Category'] = 'Cyclist Outlier'
    
    # Add categories for race visualization
    pca_df['Race_Category'] = 'Normal'
    pca_df.loc[list(race_outliers), 'Race_Category'] = 'Race Outlier'

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot normal points first
        mask_normal = (pca_df['Cyclist_Category'] == 'Normal') & (pca_df['Race_Category'] == 'Normal')
        ax.scatter(
            pca_df[mask_normal]['PC1'],
            pca_df[mask_normal]['PC2'],
            pca_df[mask_normal]['PC3'],
            c='#808080',
            label='Normal'
        )
        
        # Plot cyclist outliers if they exist
        if len(cyclist_outliers) > 0:
            mask_cyclist = pca_df['Cyclist_Category'] == 'Cyclist Outlier'
            ax.scatter(
                pca_df[mask_cyclist]['PC1'],
                pca_df[mask_cyclist]['PC2'],
                pca_df[mask_cyclist]['PC3'],
                c='#FF0000',
                label='Cyclist Outlier'
            )
        
        # Plot race outliers if they exist
        if len(race_outliers) > 0:
            mask_race = pca_df['Race_Category'] == 'Race Outlier'
            ax.scatter(
                pca_df[mask_race]['PC1'],
                pca_df[mask_race]['PC2'],
                pca_df[mask_race]['PC3'],
                c='#0000FF',
                label='Race Outlier'
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
        ax.set_title('PCA Outlier Analysis')
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        if len(cyclist_outliers) == 0 and len(race_outliers) == 0:
            print("No cyclist nor race outliers detected in the sample")
    
    # Prepare summary statistics
    summary = {
        'total_outliers': len_outliers,
        'cyclist_outliers': len(cyclist_outliers),
        'race_outliers': len(race_outliers),
        'unclassified_outliers': len(all_outliers_indx - cyclist_outliers - race_outliers),
        'top_cyclist_outliers': sorted(
            outlier_analysis['cyclist_ratios'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
        'top_race_outliers': sorted(
            outlier_analysis['race_ratios'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }
    del sampled_data
    return summary

def print_outlier_summary(summary):
    """Print a formatted summary of the outlier analysis."""
    print("\nOutlier Analysis Summary:")
    print(f"Total outliers detected: {summary['total_outliers']}")
    print(f"Cyclist-related outliers: {summary['cyclist_outliers']}")
    print(f"Race-related outliers: {summary['race_outliers']}")
    print(f"Unclassified outliers: {summary['unclassified_outliers']}")
    
    print("\nTop 10 Cyclists with highest outlier ratios:")
    for cyclist, ratio in summary['top_cyclist_outliers']:
        print(f"{cyclist}: {ratio:.2%}")
    
    print("\nTop 10 Races with highest outlier ratios:")
    for race, ratio in summary['top_race_outliers']:
        print(f"{race}: {ratio:.2%}")


def create_temporal_visualizations(data, outliers_df):
    """Create temporal visualizations for outlier analysis with improved clarity and aesthetics."""
    
    # Convert date strings to datetime
    data['date'] = pd.to_datetime(data['date'])
    outliers_df['date'] = pd.to_datetime(outliers_df['date'])
    
    # 1. Time Series Plot
    outliers_over_time = outliers_df.groupby('date').size().reset_index(name='outlier_count')
    fig_timeseries = px.line(outliers_over_time, 
                             x='date', 
                             y='outlier_count',
                             title='Outliers Over Time')
    fig_timeseries.update_layout(yaxis_title='Number of Outliers', xaxis_title='Date')
    
    # 2. Seasonal Plot with Monthly Box Plot
    outliers_over_time['month'] = outliers_over_time['date'].dt.month
    outliers_over_time['year'] = outliers_over_time['date'].dt.year
    fig_seasonal = px.box(outliers_over_time, 
                          x='month', 
                          y='outlier_count', 
                          title='Monthly Outlier Distribution Across Years',
                          labels={'month': 'Month', 'outlier_count': 'Outlier Count'})
    
    # 3. Interactive Timeline with Line and Markers
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=outliers_over_time['date'],
        y=outliers_over_time['outlier_count'],
        mode='lines+markers',
        name='Outliers',
        line=dict(color='royalblue', width=2),
        marker=dict(size=5, color='orange'),
        hovertemplate='Date: %{x}<br>Outliers: %{y}'
    ))
    fig_timeline.update_layout(title='Outlier Events Timeline', xaxis_title='Date', yaxis_title='Outlier Count')
    
    # 4. Comparative Period Analysis with Enhanced Yearly Box Plot
    yearly_comparison = px.box(outliers_over_time, 
                               x='year', 
                               y='outlier_count',
                               title='Yearly Outlier Distribution',
                               labels={'year': 'Year', 'outlier_count': 'Outlier Count'})
    
    # 5. Temporal Heatmap with Improved Readability
    heatmap_data = outliers_over_time.pivot_table(
        index='year',
        columns='month',
        values='outlier_count',
        aggfunc='sum'
    ).fillna(0)
    
    fig_heatmap = px.imshow(heatmap_data,
                            title='Outlier Count by Month and Year',
                            labels=dict(x='Month', y='Year', color='Outlier Count'),
                            color_continuous_scale='Viridis',
                            aspect='auto')
    fig_heatmap.update_layout(coloraxis_colorbar=dict(title="Count", tickvals=[0, 20, 40, 60, 80]))

    
    return {
        'timeseries': fig_timeseries,
        'seasonal': fig_seasonal,
        'timeline': fig_timeline,
        'yearly': yearly_comparison,
        'heatmap': fig_heatmap
    }



def analyze_outlier_characteristics(normal_data, outliers, feature_columns):
    """
    Analyze what makes outliers different from normal points
    
    Args:
        normal_data: DataFrame with non-outlier points
        outliers: DataFrame with outlier points
        feature_columns: List of features to analyze
    """

    # 1. Calculate z-scores for outliers
    normal_stats = normal_data[feature_columns].agg(['mean', 'std'])
    z_scores = (outliers[feature_columns] - normal_stats.loc['mean']) / normal_stats.loc['std']
    
    # 2. Find most extreme features for each outlier
    extreme_features = pd.DataFrame({
        'Feature': z_scores.abs().idxmax(axis=1),
        'Z-Score': z_scores.abs().max(axis=1)
    })
    
    # 3. Feature importance based on frequency of extreme values
    feature_importance = extreme_features['Feature'].value_counts()
    
    # 4. Distribution comparison plots with two plots per row and increased vertical space
    n_rows = (len(feature_columns) + 1) // 2  # Calculate number of rows needed for 2 plots per row
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows))  # Adjust horizontal and vertical spacing
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_columns):
        sns.kdeplot(data=normal_data[feature], ax=axes[i], label='Normal', alpha=0.5)
        sns.kdeplot(data=outliers[feature], ax=axes[i], label='Outliers', alpha=0.5)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].legend()
    
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # 5. Summary statistics
    summary = pd.DataFrame({
        'Normal_Mean': normal_data[feature_columns].mean(),
        'Normal_Std': normal_data[feature_columns].std(),
        'Outlier_Mean': outliers[feature_columns].mean(),
        'Outlier_Std': outliers[feature_columns].std(),
        'Mean_Diff_Sigma': (outliers[feature_columns].mean() - normal_data[feature_columns].mean()) / normal_data[feature_columns].std()
    })
    
    plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing between rows
    
    return {
        'extreme_features': extreme_features,
        'feature_importance': feature_importance,
        'summary_stats': summary,
        'distribution_plot': fig
    }




def plot_statistical_comparison(outlier_mean, non_outlier_mean, 
                              effect_size, outlier_std, non_outlier_std, 
                              figsize=(13, 7)):
    """
    Create plots comparing means and standard deviations for each feature.
    Each metric gets its own row with mean comparison and std deviation side by side.
    """
    metrics = outlier_mean.index
    n_metrics = len(metrics)

    fig = plt.figure(figsize=[10, 4])
    gs = fig.add_gridspec(1)
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(metrics))
    width = 0.35
    
    # 1. Effect Size Plot
    effect_bars = ax1.bar(x, effect_size, width, color='#99cc99')
    ax1.set_title('Effect Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    fig.show()
    # Create figure with subplots - one row per metric, two columns
    fig, axes = plt.subplots(n_metrics, 2, figsize=figsize, constrained_layout=True)

    
    
    # If there's only one metric, axes needs to be 2D
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
   
    x = np.array([0, 1])  # Just two bars per plot
    
    # Create plots for each metric
    for idx, metric in enumerate(metrics):
        metric_label = metric.replace('_', ' ').title()
        # Mean comparison plot
        mean_ax = axes[idx, 0]
        mean_bars1 = mean_ax.bar(x[0], outlier_mean[metric], width, 
                                label='Outlier', color='#ff9999')
        mean_bars2 = mean_ax.bar(x[1], non_outlier_mean[metric], width, 
                                label='Non-outlier', color='#66b3ff')
        mean_ax.set_title(f'{metric_label} - Mean Comparison')
        mean_ax.set_xticks(x)
        mean_ax.set_xticklabels(['Outlier', 'Non-outlier'])
        mean_ax.legend()
        mean_ax.grid(True, alpha=0.3)
            
        
        # Standard deviation plot
        std_ax = axes[idx, 1]
        std_bars1 = std_ax.bar(x[0], outlier_std[metric], width, 
                              label='Outlier', color='#3366cc')
        std_bars2 = std_ax.bar(x[1], non_outlier_std[metric], width, 
                              label='Non-outlier', color='#ff6600')
        std_ax.set_title(f'{metric_label} - Standard Deviation')
        std_ax.set_xticks(x)
        std_ax.set_xticklabels(['Outlier', 'Non-outlier'])
        std_ax.legend()
        std_ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars for both plots
        for ax in [mean_ax, std_ax]:
            bars = ax.containers[0]  # Get the bars
            ax.bar_label(bars, fmt='%.2f', padding=3)
    
    plt.suptitle('Feature Comparison: Outliers vs Non-outliers', y=1.02, fontsize=14)
    return fig

def visualize_stats(stats_dict):
    """
    Create visualization from the statistics dictionary.
    """
    fig = plot_statistical_comparison(
        outlier_mean=stats_dict['outlier_mean'],
        non_outlier_mean=stats_dict['non_outlier_mean'],
        effect_size=stats_dict['effect_size'],
        outlier_std=stats_dict['outlier_std'],
        non_outlier_std=stats_dict['non_outlier_std']
    )
    return fig


