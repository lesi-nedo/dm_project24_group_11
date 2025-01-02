import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from scipy import stats




def errors_visualization(y_true, y_pred, feature_to_predict):
    # Create figure with more height to accommodate labels
    fi = plt.figure(figsize=(30, 15))  # Increased height from 10 to 15
    
    # Create GridSpec with more space between plots
    gs = plt.GridSpec(3, 2, figure=fi, hspace=0.4)  # Added hspace parameter for vertical spacing
    
    ax1 = fi.add_subplot(gs[0, 0])
    seaborn.kdeplot(data=y_true, label='Actual', ax=ax1)
    seaborn.kdeplot(data=y_pred, label='Predicted', ax=ax1)
    ax1.set_title('Distribution of Actual vs Predicted Climb Total', pad=20)  # Added padding to title
    ax1.set_xlabel('Climb Total (meters)', labelpad=10)  # Added padding to xlabel
    ax1.set_ylabel('Density', labelpad=10)  # Added padding to ylabel
    ax1.legend()
    
    # 2. Scatter Plot with Perfect Prediction Line
    ax2 = fi.add_subplot(gs[0, 1])
    ax2.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax2.set_title(f'Actual vs Predicted {feature_to_predict.title()}', pad=20)
    ax2.set_xlabel(f'Actual {feature_to_predict.title()}', labelpad=10)
    ax2.set_ylabel(f'Predicted {feature_to_predict.title()}', labelpad=10)
    ax2.legend()

    ax3 = fi.add_subplot(gs[1, 0])
    error = y_pred - y_true
    seaborn.histplot(data=error, bins=50, ax=ax3)
    ax3.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax3.set_title('Distribution of Prediction Errors', pad=20)
    ax3.set_xlabel('Prediction Error', labelpad=10)
    ax3.set_ylabel('Count', labelpad=10)
    ax3.legend()

    # 9. Q-Q Plot of Prediction Errors
    ax4 = fi.add_subplot(gs[1, 1])
    stats.probplot(error, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot of Prediction Errors', pad=20)
    
    # 10. Error vs Actual Value
    ax5 = fi.add_subplot(gs[2, 0])
    ax5.scatter(y_true, error, alpha=0.5)
    ax5.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    ax5.set_title(f'Prediction Error vs Actual {{feature_to_predict.title()}}', pad=20)
    ax5.set_xlabel(f'Actual {feature_to_predict.title()}', labelpad=10)
    ax5.set_ylabel('Prediction Error ', labelpad=10)
    ax5.legend()
    
    # Adjust layout with extra padding


def create_prediction_visualizations(df, y_true, summary_stats, feature_name, save_path=None):
    """
    Create comprehensive visualizations for the prediction results
    Parameters:
    
    df (pd.DataFrame): DataFrame containing predictions and actual values
    summary_stats (dict): Dictionary containing summary statistics
    save_path (str, optional): Path to save the plots. If None, plots are displayed
    
    Returns:
    None
    """
    # Set up the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    y_pred = df.loc[y_true.index, f'{feature_name}_predicted']

    print(f"Length true data: {len(y_true)}")
    print(f"Length predicted data: {len(y_pred)}")

    print(f"Mean true data: {y_true.mean()}")
    print(f"Mean predicted data: {y_pred.mean()}")
    
    print(f"STD of true data: {y_true.std()}")
    print(f"STD of predicted data: {y_pred.std()}")

    errors_visualization(y_true, y_pred, feature_to_predict=feature_name)
    
    
    ax1 = fig.add_subplot(gs[0, 0])
    # 1. Distribution Usage Pie Chart
    dist_usage = pd.Series(summary_stats['distribution_usage'])
    ax1.pie(dist_usage.values, labels=dist_usage.index, autopct='%1.1f%%')
    ax1.set_title('Distribution Types Used in Predictions')
    
    # 2. Feature Importance Plot
    ax2 = fig.add_subplot(gs[0, 1])
    feature_importance = pd.Series(summary_stats['feature_importance'])
    feature_importance.sort_values().plot(kind='barh', ax=ax2)
    ax2.set_title('Feature Importance in Prediction')
    ax2.set_xlabel('Absolute Correlation with Climb Total')
        
    
    # 5. Segment Performance
    ax3 = fig.add_subplot(gs[1, 0])
    segment_performance = df.loc[y_true.index].groupby('segment', observed=True).apply(
        lambda x: np.mean(x[f'{feature_name}_predicted'] - x[feature_name].fillna(0)), include_groups = False
    ).sort_values()
    segment_performance.head(20).plot(kind='barh', ax=ax3)
    ax3.set_title('Top 20 Best Performing Segments')
    ax3.set_xlabel('Mean Absolute Error (meters)')
    
    
    
    # Adjust layout and display/save
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_segment_analysis(df, feature_name, save_path=None):
    """
    Create detailed visualizations for segment analysis
    
    Parameters:
    df (pd.DataFrame): DataFrame containing predictions and segment information
    save_path (str, optional): Path to save the plots. If None, plots are displayed
    
    Returns:
    None{'type': 'numeric'}
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    
    # 1. Segment Size Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    segment_sizes = df.groupby('segment', observed=True).size().sort_values(ascending=False)
    segment_sizes.head(20).plot(kind='bar', ax=ax1)
    ax1.set_title('Top 20 Largest Segments')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Number of Races')
    plt.xticks(rotation=45)
    
    # 2. Segment Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    segment_confidence = df.groupby('segment', observed=True)['prediction_confidence'].mean().sort_values(ascending=False)
    segment_confidence.head(20).plot(kind='bar', ax=ax2)
    ax2.set_title('Top 20 Segments by Confidence')
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Mean Confidence Score')
    plt.xticks(rotation=45)
    
    # 3. Distribution Type by Segment Size
    ax3 = fig.add_subplot(gs[1, 0])
    dist_by_size = df.groupby(['distribution_used', pd.qcut(segment_sizes, q=5)], observed=True)['segment'].count().unstack()
    if dist_by_size.isna().any().any():
        dist_by_size.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Distribution Types by Segment Size Quintile')
        ax3.set_xlabel('Distribution Type')
        ax3.set_ylabel('Number of Segments')
        plt.xticks(rotation=45)
    
    # 4. Prediction Range by Segment
    ax4 = fig.add_subplot(gs[1, 1])
    segment_ranges = df.groupby('segment', observed=True).agg({
        f'{feature_name}_predicted': ['mean', 'std']
    }).sort_values((f'{feature_name}_predicted', 'mean'), ascending=False)
    
    segment_ranges.head(20)[(f'{feature_name}_predicted', 'mean')].plot(
        kind='bar', 
        yerr=segment_ranges.head(20)[(f'{feature_name}_predicted', 'std')],
        ax=ax4
    )
    ax4.set_title('Top 20 Segments by Mean Predicted Climb')
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Mean Predicted Climb Total (meters)')
    plt.xticks(rotation=45)
    

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
most_frequent_imputer = SimpleImputer(strategy='most_frequent')
knn_imputer = KNNImputer(n_neighbors=5)
scaler = MinMaxScaler(feature_range=(-1, 1))


iterative_imputer = IterativeImputer(max_iter=40, estimator=RandomForestRegressor())

def preprocess_helper_imputer(races, pred_features ):
    missing_features = [feature for feature in pred_features if feature not in races.columns]

    if len(missing_features) > 0:
        raise ValueError(f"Missing features in the dataset: {missing_features}")


    for feature, info in pred_features.items():
        if info['type'] == 'boolean':
            feature_data = races[feature].astype(float).to_numpy().reshape(-1, 1)
            imputed_data = iterative_imputer.fit_transform(feature_data)
            races[feature] = pd.Series(imputed_data.flatten(), index=races[feature].index)
        
        elif info['type'] == 'numeric' and not feature == "points":
            feature_data = races[feature].to_numpy().reshape(-1, 1)

            non_nan_data = races[feature].dropna()
            if len(non_nan_data) > 0:
                skewness = non_nan_data.skew()
                if np.abs(skewness) > 1:
                    imputed_data = median_imputer.fit_transform(feature_data)
                else:
                    imputed_data = mean_imputer.fit_transform(feature_data)
            else:
                imputed_data = median_imputer.fit_transform(feature_data)
            
            races[feature] = pd.Series(imputed_data.flatten(), index=races[feature].index)
        elif info['type'] == 'numeric' and feature == "points":
            # get the median for each cyclist
            res = races.groupby("cyclist").agg({feature: 'mean'})
            races_feat_nan = races[races[feature].isna()]
            races.loc[races_feat_nan.index, feature] = races[['cyclist', feature]].loc[races_feat_nan.index].apply(lambda x: res.loc[x['cyclist'], feature], axis=1)
            
        elif info['type'] == 'categorical':
            feature_data = pd.Categorical(races[feature]).to_numpy().reshape(-1, 1)
            imputed_data = LabelEncoder().fit_transform(feature_data.ravel())
            imputed_data = iterative_imputer.fit_transform(imputed_data.reshape(-1, 1))
            races[feature] = pd.Series(imputed_data.flatten(), index=races[feature].index)


def preprocess_helper_scaler(races, pred_features):
    for feature, info in pred_features.items():
        if info['type'] == 'numeric':
            scaled_data = scaler.fit_transform(races[feature].to_numpy().reshape(-1, 1))
            races[feature] = pd.Series(scaled_data.flatten(), index=races[feature].index)
            
            











__all__ = [
    'create_prediction_visualizations', 
    'create_segment_analysis', 
    'errors_visualization', 
    'preprocess_helper_imputer', 
    'preprocess_helper_scaler'
]