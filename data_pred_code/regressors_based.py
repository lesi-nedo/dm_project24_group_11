import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error as rmse_func, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor


from xgboost import XGBRegressor



from .utils import errors_visualization


real_cpu_count = psutil.cpu_count(logical=False)

def compare_regressors(races_agg, feature_to_pred, features, only_positive_pred=True, round_pred=True, print_metrics=False):
    warnings.filterwarnings('ignore')
    """
    Compare different algorithms for predicting ```feature_to_pred``` values.
    
    Parameters:
    races_agg (pd.DataFrame): Input DataFrame
    features (list): List of feature columns to use
    
    Returns:
    dict: Dictionary containing trained models and their performance metrics
    """
    # Prepare data
    known_mask = ~races_agg[feature_to_pred].isna()
    train_data = races_agg[known_mask]
    predict_data = races_agg[~known_mask]
    
    X = train_data[features.keys()].copy()
    X_to_pred = predict_data[features.keys()].copy()
    y = train_data[feature_to_pred]

    for feature, info in features.items():
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
        
        if info['type'] not in ['numeric', 'categorical', 'boolean']:
            raise ValueError(f"Unknown feature type: {info['type']}")

        
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    
    
    

    models = {
        'Random Forest': RandomForestRegressor(n_jobs=real_cpu_count, n_estimators=600, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=15),
        'Gradient Boosting': GradientBoostingRegressor(subsample=0.8, n_estimators=100, min_samples_split=15, max_depth=5, learning_rate=0.05),
        'XGBoost': XGBRegressor(subsample=0.9, n_estimators=200, min_child_weight=8, max_depth=5, learning_rate=0.04, colsample_bytree=1.0),
        'Huber': HuberRegressor(max_iter = 700, epsilon = 1.5, alpha = 0.01),
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha = 2.0),
        'Lasso': Lasso(alpha= 2.0),
        'KNeighborsRegressor': KNeighborsRegressor(weights = 'distance', n_neighbors = 5, algorithm = 'ball_tree'),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(max_iter=80, max_depth=7, learning_rate=0.07, l2_regularization=0.05),
        'VotingRegressor': VotingRegressor(estimators=[
            ('rf', RandomForestRegressor(n_jobs=real_cpu_count, n_estimators=600, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=15)),
            ('gb', GradientBoostingRegressor(subsample=0.8, n_estimators=100, min_samples_split=15, max_depth=5, learning_rate=0.05)),
            ('xgb', XGBRegressor(subsample=0.9, n_estimators=200, min_child_weight=8, max_depth=5, learning_rate=0.04, colsample_bytree=1.0)),
            ('hgbr', HistGradientBoostingRegressor(max_iter=80, max_depth=7, learning_rate=0.07, l2_regularization=0.05))
        ], weights=[1, 1, 1, 1])
    }
    
    # Dictionary to store results
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
       
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=5, scoring='neg_root_mean_squared_error'
        )
        
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        if round_pred:
            y_pred = np.round(y_pred)
        
        if only_positive_pred:
            y_pred = np.maximum(y_pred, 0)
        
        rmse = rmse_func(y_test, y_pred)
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'cv_rmse': -cv_scores.mean(),
            'cv_rmse_std': cv_scores.std(),
            'predictions': y_pred
        }
        if print_metrics:
            print(f"{name} Results:")
            print(f"R²: {r2:.3f}")
            print(f"CV RMSE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} meters")
    
    # Visualize results
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot RMSE comparison
    rmse_scores = [(name, results[name]['rmse']) for name in results.keys()]
    rmse_scores.sort(key=lambda x: x[1])
    
    names, scores = zip(*rmse_scores)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(names, scores)
    ax1.set_xlabel('RMSE (meters)')
    ax1.set_title('Model Comparison - RMSE (lower is better)')
    
    # Plot R² comparison
    r2_scores = [(name, results[name]['r2']) for name in results.keys()]
    r2_scores.sort(key=lambda x: x[1], reverse=True)
    
    names, scores = zip(*r2_scores)
    ax2 = fig.add_subplot(gs[0, 1])

    plt.barh(names, scores)
    ax2.set_xlabel('R² Score (higher is better)')
    ax2.set_title('Model Comparison - R² Score')
    
    # Plot residuals for best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['cv_rmse'])
    best_predictions = results[best_model_name]['predictions']
    
    ax3 = fig.add_subplot(gs[1, :])
    residuals = y_test - best_predictions
    ax3.scatter(best_predictions, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'Residuals Plot - {best_model_name}')
    
    plt.show()
    plt.close(fig=fig)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'R²': [results[name]['r2'] for name in results.keys()],
        'CV RMSE': [results[name]['cv_rmse'] for name in results.keys()],
        'CV RMSE Std': [results[name]['cv_rmse_std'] for name in results.keys()]
    }, index=results.keys())
    
    print("\nSummary of all models:")
    print(summary.sort_values('CV RMSE'))

    feature_predictions = pd.Series(results[best_model_name]['model'].predict(X_to_pred), index=predict_data.index)
    
    print("Visualizations for the best model:")
    errors_visualization(y_test, best_predictions, feature_to_pred)

    warnings.filterwarnings('default')
    return  results, summary, feature_predictions


__all__ = ['compare_regressors']