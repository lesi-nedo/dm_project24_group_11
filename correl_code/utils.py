import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from plotly.subplots import make_subplots



def create_correlation_matrix(merged_df, columns_of_interest):
    # Select the relevant columns from the correlation matrix shown in the image
    
    
    # Create a copy of races dataframe with selected columns
    race_data = merged_df[columns_of_interest].copy()
    
    # Calculate correlation matrix
    corr_matrix = race_data.corr(method='pearson')
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    seaborn.heatmap(corr_matrix, 
                annot=True,  # Show correlation values
                fmt='.2f',   # Format to 2 decimal places
                cmap='RdGy_r',  # Red-Grey-Blue colormap reversed
                vmin=-1,     # Minimum correlation value
                vmax=1,      # Maximum correlation value
                center=0,    # Center the colormap at 0
                square=True) # Make the plot square-shaped
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add title
    plt.title('Correlation Matrix: Race vs Cyclist Features')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf(), corr_matrix


def display_top_correlations(corr_matrix, n=6):
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    
    # Stack the correlations and sort by absolute value
    stacked = upper.stack()
    sorted_corr = stacked.loc[stacked.sort_values(ascending=False).index]
    correlations = {
        "Positive": {
            "Features Pair": [],
            "Correlation": []
        },
        "Negative": {
            "Features Pair": [],
            "Correlation": []
        }
    }
    for i in range(min(n, len(sorted_corr))):
        # pair = sorted_corr.index[i]
        # value = sorted_corr.iloc[i]
        # print(f"{pair[0]} vs {pair[1]}: {value:.3f}")
        correlations["Positive"]["Features Pair"].append(f"{sorted_corr.index[i][0]} vs {sorted_corr.index[i][1]}")
        correlations["Positive"]["Correlation"].append(sorted_corr.iloc[i])
        correlations["Negative"]["Features Pair"].append(f"{sorted_corr.index[-(i+1)][0]} vs {sorted_corr.index[-(i+1)][1]}")
        correlations["Negative"]["Correlation"].append(sorted_corr.iloc[-(i+1)])
    
    return pd.DataFrame(correlations["Positive"]), pd.DataFrame(correlations["Negative"])
        
def print_top_positive_correlations(df, message="Top Positive Correlations"):
    # Set color palette


    # Increase the figure size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot the data
    seaborn.barplot(
        data=df,
        y='Features Pair',
        x='Correlation',
        ax=ax,
        hue='Features Pair',
        dodge=False,
        legend=False
    )

    # Customize the plot
    ax.set_title(message, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient', fontsize=14)
    ax.set_ylabel('Feature Pairs', fontsize=14)

    # Add correlation value labels to the bars
    for i, v in enumerate(df['Correlation']):
        ax.text(v * 0.95, i, f'{v:.3f}', va='center', ha='right', color='black', fontsize=10)

    # Customize grid and appearance
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    # Set x-axis limits for positive values
    ax.set_xlim(0, df['Correlation'].max() * 1.1)

    # Remove top and right spines
    seaborn.despine()

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.close()


def print_top_negative_correlations(df, message="Top Negative Correlations"):
    # Set color palette

    # Increase the figure size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot the data
    seaborn.barplot(
        data=df,
        y='Features Pair',
        x='Correlation',
        ax=ax,
        hue='Features Pair',
        dodge=False,
        legend=False
    )

    # Customize the plot
    ax.set_title(message, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient', fontsize=14)
    ax.set_ylabel('Feature Pairs', fontsize=14)

    # Add correlation value labels to the bars, slightly shifted inward for negative values
    for i, v in enumerate(df['Correlation']):
        ax.text(v * 1.05, i, f'{v:.3f}', va='center', ha='left', color='black', fontsize=10)

    # Customize grid and appearance
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    # Set x-axis limits for negative values
    ax.set_xlim(df['Correlation'].min() * 1.1, 0)

    # Remove top and right spines
    seaborn.despine()

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.close()

def calc_group_correlations(data, group_column):
    correlations = {}
    for group in data[group_column].unique():
        group_data = data[data[group_column] == group]
        if len(group_data) > 1:  # Ensure we have enough data points
            if group_data['average_temperature'].nunique() > 1 and group_data['average_temperature'].std() > 0 and group_data['points'].std() > 0:  # Ensure temperature varies
                corr, _ = stats.pearsonr(group_data['average_temperature'], group_data['points'])
                correlations[group] = corr
    return pd.Series(correlations)


def analyze_cyclist_progression(cyclist_name, data, min_races=10, plot=True):
    cyclist_data = data[data['cyclist'] == cyclist_name].sort_values('date')
    
    if len(cyclist_data) < min_races:
        #print(f"{cyclist_name} has fewer than {min_races} races. Skipping.")
        return None
    
    # Calculate a rolling average of points
    cyclist_data['rolling_avg_points'] = cyclist_data['points'].rolling(window=10, min_periods=1).mean()
    if plot:
        # Plot the first
        # Plot career progression
        plt.figure(figsize=(12, 6))
        plt.plot(cyclist_data['date'], cyclist_data['rolling_avg_points'], label='Rolling Avg Points (10 races)')
        plt.scatter(cyclist_data['date'], cyclist_data['points'], alpha=0.3, color='gray', label='Race Points')
        plt.title(f"Career Progression of {cyclist_data['name_x'].iloc[0]}")
        plt.xlabel('Date')
        plt.ylabel('Points')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    # Correlation between age and performance
    nona_data = cyclist_data.dropna(subset=['cyclist_age', 'points'])
    if nona_data['cyclist_age'].nunique() > 1 and nona_data['points'].std() > 0 \
        and nona_data['cyclist_age'].std() > 0:
        correlation, p_value = stats.pearsonr(nona_data['cyclist_age'], nona_data['points'])
    else:
        correlation, p_value = None, None
    return {
        'name': cyclist_data['name'].iloc[0],
        'cyclist': cyclist_name,
        'num_races': len(cyclist_data),
        'career_span': (cyclist_data['date'].max() - cyclist_data['date'].min()).days / 365,
        'avg_points': cyclist_data['points'].mean(),
        'max_points': cyclist_data['points'].max(),
        'age_performance_correlation': correlation,
        'p_value': p_value
    }

def analyze_performance_peaks(cyclist_name, data, min_races=10):
    cyclist_data = data[data['cyclist'] == cyclist_name].sort_values('date')
    
    if len(cyclist_data) < min_races:
        # print(f"{cyclist_name} has fewer than {min_races} races. Skipping.")
        return None
    
    # Find peak performance year
    peak_year = cyclist_data.groupby('year')['points'].mean().idxmax()
    peak_age = peak_year - cyclist_data['birth_year'].iloc[0]
    
    return {
        'name': cyclist_data['name'].iloc[0],
        'cyclist': cyclist_name,
        'peak_year': peak_year,
        'peak_age': peak_age,
        'peak_points': cyclist_data[cyclist_data['year'] == peak_year]['points'].mean()
    }

# Function to create enhanced radar plot using Plotly
def radar_plot(data, title, attributes):
    # Calculate grid dimensions
    n_profiles = len(data.keys())
    profiles = list(data.keys())
    n_cols = min(5, n_profiles)  # Max 3 columns
    n_rows = (n_profiles + n_cols - 1) // n_cols

    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Profile {p}" for p in profiles],
        specs=[[{'type': 'polar'}]*n_cols for _ in range(n_rows)],
        vertical_spacing=0.01
    )

    # Add traces for each profile
    for idx, profile in enumerate(profiles):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        
        fig.add_trace(
            go.Scatterpolar(
                r=data[profile],
                theta=attributes + [attributes[0]],
                fill='toself',
                name=f"{profile}",
                opacity=0.6,
                marker=dict(size=8),
                connectgaps=True,
            ),
            row=row, col=col
        )
            

    # Update layout
    fig.update_layout(
        height=450*n_rows,  # Adjust height based on number of rows
        width=1200,         # Fixed width
        title=dict(
            text=title,
            y=1.0,
            x=0.0,
            xanchor='center',
            yanchor='top'
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=10),
            yanchor="top",
            y=0.5,
            xanchor="right",
            x=1.0
        ),
        margin=dict(t=100, b=5, l=50, r=5)
    )

    # Update each polar subplot
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            showline=True,
            linewidth=1,
            tickfont=dict(size=10),
        ),
        
    )
    
    return fig


def analyze_correlation_differences(corr_matrix1, corr_matrix2, columns_of_interest):
    """
    Analyze and visualize differences between two correlation matrices.
    
    Parameters:
    corr_matrix1: First correlation matrix (with outliers)
    corr_matrix2: Second correlation matrix (without outliers)
    columns_of_interest: List of column names
    """
    # Calculate absolute differences between matrices
    diff_matrix = np.abs(corr_matrix1 - corr_matrix2)
    
    # Create a DataFrame for easier handling
    diff_df = pd.DataFrame(diff_matrix, 
                          index=columns_of_interest,
                          columns=columns_of_interest)
    
    # Get the top differences (excluding diagonal)
    np.fill_diagonal(diff_matrix, 0)  # Zero out diagonal
    
    # Find the largest differences
    flat_indices = np.argwhere(diff_matrix == np.max(diff_matrix))[0]
    max_diff = diff_matrix[flat_indices[0], flat_indices[1]]
    feature1 = columns_of_interest[flat_indices[0]]
    feature2 = columns_of_interest[flat_indices[1]]
    
    # Create heatmap of differences
    plt.figure(figsize=(12, 10))
    sns.heatmap(diff_df, 
                annot=True, 
                cmap='YlOrRd',
                fmt='.2f',
                center=0)
    plt.title('Absolute Differences Between Correlation Matrices')
    
    return {
        'max_difference': max_diff,
        'feature_pair': (feature1, feature2),
        'diff_matrix': diff_df
    }

def compare_correlations(df_with_outliers, df_without_outliers, columns_of_interest):
    """
    Compare correlation matrices for datasets with and without outliers.
    """
    # Calculate correlation matrices
    corr_with_outliers = df_with_outliers[columns_of_interest].corr()
    corr_without_outliers = df_without_outliers[columns_of_interest].corr()
    
    # Analyze differences
    results = analyze_correlation_differences(
        corr_with_outliers.values,
        corr_without_outliers.values,
        columns_of_interest
    )
    
    return results


__all__ = [
    'create_correlation_matrix',
    'display_top_correlations',
    'print_top_positive_correlations',
    'print_top_negative_correlations',
    'calc_group_correlations',
    'analyze_cyclist_progression',
    'analyze_performance_peaks',
    'radar_plot',
    'compare_correlations'

]