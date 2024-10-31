
import psutil
import multiprocessing as mp



import pandas as pd

import matplotlib.pyplot as plt

import seaborn
import numpy as np
import warnings




from sklearn.metrics import root_mean_squared_error as mse
from sklearn.metrics import r2_score, silhouette_score

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity



from scipy.stats import kstest
from scipy import stats


from tqdm import tqdm

from pandarallel import pandarallel



pandarallel.initialize(progress_bar=True)
real_cpu_count = psutil.cpu_count(logical=False)




tqdm.pandas()  # Enable progress_apply


def add_jitter(data, epsilon=1e-9):
    return data + np.random.normal(0, epsilon, len(data))

def fit_best_distribution(data):
    """
    Try multiple distributions and return the best fitting one
    
    Parameters:
    data (array-like): Data to fit distributions to
    
    Returns:
    tuple: (distribution name, distribution parameters, fit score)
    """
    orig_mean = data.mean()
    orig_std = data.std()
    orig_skew = 0
    orig_kurtosis = 0
    kde = None
    if orig_std > 1e-4:
        orig_skew = stats.skew(add_jitter(data))
        orig_kurtosis = stats.kurtosis(data)
        kde = stats.gaussian_kde(data)
    
        


    distributions = [
        ('gamma', stats.gamma),
        ('lognorm', stats.lognorm),
        ('weibull_min', stats.weibull_min),
        ('beta', stats.beta),
        ('burr', stats.burr),
        ('burr12', stats.burr12),
        ('gaussian_mix', lambda x: stats.norm(loc=orig_mean, scale=orig_std))
    ]
    
    best_fit = None
    best_score = float('inf')
    
    if len(data) < 20 or data.std() == 0:
        return None
    
    
    
    for name, distribution in distributions:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == 'beta':
                    data_normalized = (data - data.min()) / (data.max() - data.min())
                    data_normalized = np.clip(data_normalized, 1e-6, 1 - 1e-6)
                    params = distribution.fit(data_normalized)
                    ks_stat, _ = kstest(data_normalized, name, params)
                    samples = distribution.rvs(*params, size=1000)
                    samples = samples * (data.max() - data.min()) + data.min() 
                else:
                    params = distribution.fit(data)
                    ks_stat, _ = kstest(data, name, params)
                    samples = distribution.rvs(*params, size=1000)

                
                moment_score = (
                    abs(orig_mean - np.mean(samples)) / orig_std +
                    abs(orig_std - np.std(samples)) / orig_std +
                    abs(orig_skew - stats.skew(samples)) +
                    abs(orig_kurtosis - stats.kurtosis(samples))
                )

                weight_ks = 1.1  # KS statistic weight
                weight_moment = 0.8  # Moment score weight
                weight_density = 0.5  # Density score weight

                density_score = 0
                if kde is not None:
                    x_eval = np.linspace(data.min(), data.max(), len(data))
                    kde_orig =  kde(x_eval)
                    kde_fitted = stats.gaussian_kde(samples)(x_eval)
                    density_score = np.mean(np.abs(kde_orig - kde_fitted))


                total_score = (weight_ks * ks_stat) + (weight_moment * moment_score) + (weight_density * density_score)
                
                if total_score < best_score:
                    best_fit = (name, params, total_score)
                    best_score = ks_stat
                    
        except Exception as e:
            continue
    
    return best_fit



def find_optimal_clusters(data, max_k, fig, gs):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    # Plot inertia vs. number of clusters
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, max_k + 1), inertia, marker='o')
    ax1.set_title('Elbow Method For Optimal k')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia (Sum of squared distances)')

def silhouette_method(data, max_k, fig, gs):
    silhouette_scores = []
    for k in range(2, max_k + 1):  # Silhouette requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    # Plot silhouette score vs. number of clusters
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Method For Optimal k')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')


def parallel_process_segments(segments, df, feature_to_predict, segmentation_features):
    segment_distributions = {}
    distribution_info = {}

    for segment in tqdm(segments, desc="Fitting distributions", total=len(segments)):
        segment_data = df[df['segment'] == segment][feature_to_predict].dropna()
        if len(segment_data) > 20:
            best_fit = fit_best_distribution(segment_data)

            if best_fit is not None:
                dist_name, params, ks_score = best_fit
                segment_distributions[segment] = {
                    "distribution": (dist_name, params),
                    "data": df[df['segment'] == segment][feature_to_predict],
                }
                distribution_info[segment] = {
                    'distribution': dist_name,
                    'ks_score': ks_score,
                    'sample_size': len(segment_data),
                    'feature_means': {
                        feature: df[df['segment'] == segment][feature].mean()
                        for feature in segmentation_features if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
                    }
                }
            else:
                segment_distributions[f"{segment}_none"] = df[df['segment'] == segment][segmentation_features.keys()].to_numpy().reshape(-1, len(segmentation_features))
                distribution_info[segment] = {
                    'distribution': 'none',
                    'ks_score': 1.0,
                    'sample_size': len(segment_data),
                    'feature_means': {
                        feature: df[df['segment'] == segment][feature].mean()
                        for feature in segmentation_features if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
                    }
                }
        else:
            segment_distributions[f"{segment}_none"] = df[df['segment'] == segment][segmentation_features.keys()].to_numpy().reshape(-1, len(segmentation_features))

    return {"segments_dist": segment_distributions, "distribution_info": distribution_info}

def predict_feat(row,  kwargs):
    
    data_mean = kwargs.get('data_mean')
    data_std = kwargs.get('data_std')
    segment_distributions = kwargs.get('segment_distributions')
    distribution_info = kwargs.get('distribution_info')
    segmentation_features = kwargs.get('segmentation_features')
    feature_to_predict = kwargs.get('feature_to_predict')


    min_valid = max(0, data_mean - 4 * data_std)
    max_valid = data_mean + 4 * data_std
    if pd.isna(row[feature_to_predict]):
        segment = row['segment']


        if segment in segment_distributions:
            try:
                dist_name, params = segment_distributions[segment]['distribution']
                segment_data = segment_distributions[segment]['data']

                distribution = getattr(stats, dist_name)

                with np.errstate(all='ignore'):  # Suppress numpy warnings
                    if dist_name == 'beta':
                        segment_data_no_na = segment_data.dropna()

                        min_val, max_val = segment_data_no_na.min(), segment_data_no_na.max()
                        samples = distribution.rvs(*params, size=100)
                        prediction = np.median(samples) * (max_val - min_val) + min_val
                    else:
                        try:
                            samples = distribution.rvs(*params, size=100)
                            prediction = np.mean(samples)
                        except:
                            prediction = distribution.mean(*params)
                    if np.isnan(prediction) or np.isinf(prediction) or prediction < min_valid or prediction > max_valid:
                        if len(segment_data) > 0:
                            prediction = segment_data.mean() + np.random.normal(0, segment_data.std() * 0.005)
                        else:
                            prediction = data_mean + np.random.normal(0, data_std * 0.005)

                

                return np.clip(prediction, min_valid, max_valid)
            except:
                print(f"Error predicting segment {segment}")
                return data_mean + np.random.normal(0, data_std * 0.005)

       
        
        valid_segments = [s for s in segment_distributions.keys() 
                        if s in distribution_info]
        if not valid_segments:
            return data_mean + np.random.normal(0, data_std * 0.005)
        
        segment_means = np.array([
            [distribution_info[s]['feature_means'].get(f, 0) 
            for f in segmentation_features]
            for s in valid_segments
        ])

        segment_features = segment_distributions[f"{segment}_none"]
        similarities = cosine_similarity(segment_features, segment_means).flatten()
        
        top_segments = np.argsort(similarities)[-3:]  # Top 3 most similar segments
        
        predictions = []
        for idx in top_segments:
            idx = np.unravel_index(idx, (segment_features.shape[0], segment_means.shape[0]))[1]
            seg = valid_segments[idx]
            dist_name, params = segment_distributions[seg]['distribution']
            distribution = getattr(stats, dist_name)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                pred = distribution.mean(*params)
                
            if not (np.isnan(pred) or np.isinf(pred)) and min_valid <= pred <= max_valid:
                predictions.append(pred)
        
        if predictions:
            return np.mean(predictions) + np.random.normal(0, data_std * 0.005)
    
    return row[feature_to_predict]

def predict_feat_in_chunks(rows, kwargs):
    results_series = pd.Series(dtype='float64')
    for idx, row in rows.iterrows():
        results_series.at[idx] = predict_feat(row, kwargs)
    
    return results_series
        
        

def predict_feature_density(df, segmentation_features, feature_to_predict, n_clusters_final=0, run_silhouette=True, plot_clusters=True):
    """
    Predict missing a specific feature nan values using data segmentation and multiple probability distributions
    
    Parameters:
    races_df (pd.DataFrame): The input races dataframe
    
    Returns:
    pd.DataFrame: DataFrame with predicted feature to predict values and fit information
    """
    
    with mp.Pool() as pool:

        missing_features = [k for k,v in segmentation_features.items() if k not in df.columns]
        

        if missing_features:
            raise ValueError(f"Features {missing_features} are missing in the input DataFrame")
            
        
        
        fig = plt.figure(figsize=(10, 6))
        gs = plt.GridSpec(2, 2, figure=fig, wspace=0.5, hspace=0.5)

        
        # 1. Feature preparation
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        print("\n Feature statistics before imputation:")
        for feature in segmentation_features:
            missing = df[feature].isna().sum()
            total = len(df[feature])
            print(f"{feature}: {missing} missing ({missing/total:.2%})")
        
        # 2. Create feature matrix for segmentation
        feature_matrix = df[segmentation_features.keys()].to_numpy()
        
        if np.isnan(feature_matrix).any():
            raise ValueError("NaN values found in feature matrix")
        
        # 3. Use K-means clustering for sophisticated segmentation
        n_clusters = min(20, (len(df) // 1000))+10  # Adaptive number of clusters
        find_optimal_clusters(feature_matrix, n_clusters, fig, gs)
        if run_silhouette:
            silhouette_method(feature_matrix, n_clusters, fig, gs)
        n_clusters = 8  # Chosen based on elbow and silhouette methods
        kmeans = KMeans(n_clusters=n_clusters_final | n_clusters, random_state=42)
        df['segment_cluster'] = kmeans.fit_predict(feature_matrix)
        
        if plot_clusters:
            # plot the clusters
            ax3 = fig.add_subplot(gs[1, 0])
            seaborn.scatterplot(x=feature_matrix[:, 0], y=feature_matrix[:, 1], hue=df['segment_cluster'], s=50, palette='viridis', ax=ax3)
            ax3.set_title('K-Means Clustering of Segmentation Features')
            ax3.set_xlabel('Feature 1')
            ax3.set_ylabel('Feature 2')
            ax3.legend(title='Cluster')
            

            centers = kmeans.cluster_centers_
            ax4 = fig.add_subplot(gs[1, 1])
            seaborn.scatterplot(x=centers[:, 0], y=centers[:, 1], s=100, color='red', marker='X', ax=ax4)
            ax4.set_title('Cluster Centers')
            

        
        plt.show()
        plt.close(fig)
        # 4. Create segments
        df['length_category'] = pd.qcut(df['length'], q=5, labels=['VS', 'S', 'M', 'L', 'VL'])
        df['road_type'] = np.where(df['is_tarmac'], 'T', 'M')
        df['season_seg'] = pd.cut(df['month'], bins=[0, 4, 8, 12], labels=['Spring', 'Summer', 'Fall'])
        
        if 'startlist_quality' in df.columns:
            df['quality_level'] = np.where(
                df['startlist_quality'] > df['startlist_quality'].median(), 
                'High', 
                'Low'
            )
        else:
            df['quality_level'] = 'Medium'
        
        # Create segment ID efficiently
        df['segment'] = (df['length_category'].astype(str) + '_' + 
                        df['road_type'].astype(str) + '_' + 
                        df['season_seg'].astype(str) + '_' + 
                        df['quality_level'].astype(str) + '_C' + 
                        df['segment_cluster'].astype(str))
        
        print(f"number of segments: {len(df['segment'].unique())}")
        # 5. Calculate distributions for each segment
        segment_distributions = {}
        distribution_info = {}        

        segments_each_process = np.array_split(df['segment'].unique(), int(real_cpu_count * 0.3))
        info=  pool.starmap(parallel_process_segments, [(segments, df, feature_to_predict, segmentation_features) for segments in segments_each_process], chunksize=2)
        for i in info:
            segment_distributions.update(i['segments_dist'])
            distribution_info.update(i['distribution_info'])
            
        # 6. Predict missing values
        df_chunks = np.array_split(df,  int(real_cpu_count * 0.25))
        print(f"number of chunks: {len(df) // len(df_chunks)}")

        kwargs = {
            "feature_to_predict": feature_to_predict,
            "segment_distributions": segment_distributions,
            "distribution_info": distribution_info,
            "segmentation_features": segmentation_features,
            "feature_matrix": feature_matrix,
            "data_mean": df[feature_to_predict].mean(),
            "data_std": df[feature_to_predict].std(),
            "all_segments": df['segment'].unique()

        }
        
        df[f'{feature_to_predict}_predicted'] = pd.concat(
            pool.starmap(predict_feat_in_chunks, 
                         [(rows, kwargs) for rows in df_chunks], chunksize=2
            )).round()

        

        # 7. Calculate prediction confidence
        def calculate_confidence(row):
            segment = row['segment']
            if segment in distribution_info:
                info = distribution_info[segment]
                ks_confidence = 1 / (1 + info['ks_score'])
                sample_confidence = min(1, info['sample_size'] / 100)
                feature_similarity = np.mean([info['feature_means'].get(f, 0) for f in segmentation_features])
                return (ks_confidence + sample_confidence + feature_similarity) / 3
            return 0
        
        df['prediction_confidence'] = df.progress_apply(calculate_confidence, axis=1)
        # df['segment'].map(lambda x: print(x) if distribution_info.get(x, {}) == {} else None)
        df['distribution_used'] = df['segment'].map(lambda x: distribution_info.get(x, {}).get('distribution', 'none'))
        
        # 8. Create summary statistics
        summary_stats = {
            'total_missing': df[feature_to_predict].isna().sum(),
            'segments_created': len(df['segment'].unique()),
            'segments_with_distribution': len(segment_distributions),
            'distribution_usage': pd.Series([info['distribution'] for info in distribution_info.values()]).value_counts().to_dict(),
            'mean_confidence': df['prediction_confidence'].mean(),
            'median_predicted_climb': df[f'{feature_to_predict}_predicted'].median(),
            'features_used': segmentation_features,
            'feature_importance': {
                feature: abs(np.corrcoef(feature_matrix[:, i], 
                                    df[feature_to_predict].fillna(df[feature_to_predict].mean()))[0, 1])
                for i, feature in enumerate(segmentation_features)
            }
        }
        del feature_matrix
        return df[[feature_to_predict, f'{feature_to_predict}_predicted', 'prediction_confidence', 
              'segment', 'distribution_used']], summary_stats

    

def print_density_info(stats_races, predictions, y_test, feature_to_predict):

    print("\nMetrics:")
    mse_score = mse(y_test, predictions[f'{feature_to_predict}_predicted'].loc[y_test.index])
    r2 = r2_score(y_test, predictions[f'{feature_to_predict}_predicted'].loc[y_test.index])
    rmse = np.sqrt(mse_score)
    mae = np.mean(np.abs(y_test - predictions[f'{feature_to_predict}_predicted'].loc[y_test.index]))
    print(f"RMSE: {rmse:.2f} meters")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.2f} meters")
    print(f"MSE: {mse_score:.2f}")


__all__ = ['predict_feature_density', 'print_density_info']