from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np


def check_if_numerical(df: pd.DataFrame, features: list[str]) -> set[str]:
    """
    Check for numerical columns in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    
    features : list
        List of features to check for numerical columns
        
    Returns:
    --------
    list
        List of numerical columns
    """
    df_num_col = df.select_dtypes(include='number').columns.tolist()

    return set(features) - set(df_num_col)

def normalize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Normalize features in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to normalize
        
    features : list
        List of features to normalize
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized features
    """
    df_normalized = df[features].copy()
    df_normalized = StandardScaler().fit_transform(df_normalized)
    
    return df_normalized



class FeaturesPCAReducer:
    def __init__(self, n_components=None, variance_threshold=0.95):
        
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components

        self.explained_variance_ = None
        self.components_ = None
        self.variance_threshold = variance_threshold
        

        
    def combine_features(self, df, features, plot=True):
        """
        Combines `features` using PCA.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing `features` columns
        features : list
            List of features to combine
        plot : bool
            Whether to create explanation plots
            
        Returns:
        --------
        pandas.Series
            Combined features using first principal component
        """

        # Check if features are numerical
        non_numerical_features = check_if_numerical(df, features)
        if non_numerical_features:
            raise ValueError(f"Features {non_numerical_features} are not numerical")
        
        # Normalize the features
        scaled_data = normalize_features(df, features)
        
        # Apply PCA
        transformed_data = self.pca.fit_transform(scaled_data)
        
        # Store explained variance and components for analysis
        self.cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
        self.components_ = self.pca.components_
        self.explained_variance_ = self.pca.explained_variance_ratio_
        
        if self.n_components is None:
            self.n_components = np.argmax(self.cumulative_variance >= self.variance_threshold) + 1
            self.pca = PCA(n_components=self.n_components)
            transformed_data = self.pca.fit_transform(scaled_data)
        
        transformed_df = pd.DataFrame(
            transformed_data,
            index=df.index,
            columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])]
        )

        if plot:
            self._create_analysis_plots(df[features], transformed_data)
        
        return transformed_df
    
    def _create_analysis_plots(self, original_data, transformed_data):
        """
        Creates explanatory plots for the PCA transformation
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            Original points and UCI points data
        transformed_data : numpy.ndarray
            PCA transformed data
        """
        n_components = transformed_data.shape[1]
        
        # Calculate number of rows and columns needed for PC scatter plots
        n_pairs = (n_components * (n_components - 1)) // 2
        # if n_pairs > 0:
        #     scatter_rows = min(3, np.ceil(n_pairs / 2))
        #     scatter_cols = min(2, np.ceil(n_pairs / scatter_rows))
        
        # Create subplots with dynamic layout
        n_plots = 2 + (1 if n_pairs > 0 else 0)  # Original + variance + scatters
        fig = plt.figure(figsize=(20, (5 * np.ceil(n_plots/2)).astype(int)))
        
        # Plot 1: Original points distribution with correlation line
        ax1 = plt.subplot(np.ceil(n_plots/2).astype(int), 2, 1)
        seaborn.scatterplot(
            data=original_data,
            x=original_data.columns[0],
            y=original_data.columns[1],
            alpha=0.5,
            ax=ax1
        )
        seaborn.regplot(
            data=original_data,
            x=original_data.columns[0],
            y=original_data.columns[1],
            scatter=False,
            color='red',
            ax=ax1
        )
        ax1.set_title('Original Data Distribution')
        
        # Plot 2: Explained variance
        ax2 = plt.subplot(np.ceil(n_plots/2).astype(int), 2, 2)
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_components)],
            'Explained Variance': self.explained_variance_
        })
        seaborn.barplot(
            data=variance_df,
            x='Component',
            y='Explained Variance',
            ax=ax2
        )
        ax2.set_title('Explained Variance by Component')
        ax2.set_ylim(0, 1)
        
        # Add percentage annotations on bars
        for i, v in enumerate(self.explained_variance_):
            ax2.text(
                i,
                v + 0.01,
                f'{v*100:.1f}%',
                ha='center'
            )
        
        # Create scatter plots for PC pairs if we have multiple components
        if n_pairs > 0:
            plot_idx = 3
            for i in range(n_components):
                for j in range(i+1, n_components):
                    if plot_idx <= np.ceil(n_plots/2) * 2:
                        ax = plt.subplot(np.ceil(n_plots/2).astype(int), 2, plot_idx)
                        seaborn.scatterplot(
                            x=transformed_data[:, i],
                            y=transformed_data[:, j],
                            alpha=0.5,
                            ax=ax
                        )
                        ax.set_title(f'PC{i+1} vs PC{j+1}')
                        ax.set_xlabel(f'Principal Component {i+1}')
                        ax.set_ylabel(f'Principal Component {j+1}')
                        
                        # Add direction vectors
                        scale = np.std(transformed_data[:, [i,j]]) * 0.5
                        ax.arrow(
                            0, 0,
                            self.components_[i,0] * scale,
                            self.components_[i,1] * scale,
                            color='r',
                            alpha=0.5,
                            head_width=scale*0.1
                        )
                        ax.arrow(
                            0, 0,
                            self.components_[j,0] * scale,
                            self.components_[j,1] * scale,
                            color='r',
                            alpha=0.5,
                            head_width=scale*0.1
                        )
                        plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed component information
        print("\nPCA Components Information:")
        cumulative_var = np.cumsum(self.explained_variance_)
        
        print("\nExplained Variance Ratio:")
        for i, var in enumerate(self.explained_variance_):
            print(f"PC{i+1}: {var*100:.1f}% (Cumulative: {cumulative_var[i]*100:.1f}%)")
        
        print("\nComponent Weights:")
        feature_names = original_data.columns
        for i in range(n_components):
            print(f"\nPC{i+1}:")
            for j, feature in enumerate(feature_names):
                print(f"  {feature}: {self.components_[i,j]:.3f}")
        
        # Additional statistics for transformed data
        print("\nTransformed Data Statistics:")
        pc_df = pd.DataFrame(
            transformed_data,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        print(pc_df.describe())

def combine_race_points(df, add_to_df=True, column_name='pca_race_points', plot_analysis=True):
    """
    Wrapper function to combine points using PCA
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'points' and 'uci_points' columns
    plot_analysis : bool
        Whether to show analysis plots
        
    Returns:
    --------
    DataFrame with original and combined points
    """
    # Create and apply PCA reducer
    reducer = FeaturesPCAReducer(n_components=2)
    combined_points = reducer.combine_features(
        df,
        ['points', 'uci_points'],
        plot=plot_analysis
    )['PC1']    
    # Add combined points to dataframe
    if add_to_df:
        df[column_name] = combined_points
        return df
    else:
        result_df = df.copy()
        result_df[column_name] = combined_points
        return result_df
    


__all__ = ['combine_race_points']


if __name__ == '__main__':
    print("Running PCA Reducer")
    # Load data
    df = pd.read_csv('dataset/races_filled.csv')

    # Combine points using PCA
    df = combine_race_points(df, add_to_df=True, plot_analysis=True)

    