import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error as mse, r2_score

from typing import Tuple, Optional, Dict

from .utils import errors_visualization



def predict_feature_knnimp(
    df: pd.DataFrame,
    features: Dict[str, Dict[str, str]],
    feature_to_predict: str,
    type: str,
    y_true: pd.Series | None = None,
    n_neighbors: int = 5,
    min_samples_required: int = 100,
    return_metrics: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Predicts missing climb_total values in cycling race data using KNN imputation
    with advanced feature engineering and validation.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        # Validate input data
        missing_race_cols = [col for col in features if col not in df.columns]
        if missing_race_cols:
            raise ValueError(f"Missing required columns. Races: {missing_race_cols}")

        logger.info("Starting prediction process...")
        
        
        
        features = {**features, feature_to_predict: {'type': type}}
        
        X = df[features.keys()].copy()
        
        if len(X) < min_samples_required:
            raise ValueError(f"Insufficient data: {len(X)} samples, {min_samples_required} required")
        
        
        
        # Apply KNN imputation
        logger.info("Applying KNN imputation...")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
        X_imputed = imputer.fit_transform(X)
        
    
        
        # Extract only the `feature_to_predict`` column from predictions
        feature_to_pred_idx = list(X.columns).index(feature_to_predict)
        predicted_feature = X_imputed[:, feature_to_pred_idx]

        y_pred = pd.Series(predicted_feature, index=X.index, name=f'{feature_to_predict}_predicted').round()
        
        
        # Calculate metrics if requested
        metrics = None
        if return_metrics and y_true is not None:
            logger.info("Calculating validation metrics...")
            y_pred = y_pred.loc[y_true.index]  # Ensure we're comparing 1D arrays 
            logger.info(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")

            
            mse_con = mse(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse_con)
            mae = np.mean(np.abs(y_true - y_pred))
            
            metrics = {
                'mse': round(mse_con, 2),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2': round(r2, 2),
                'samples_used': len(y_pred),
            }
            errors_visualization(y_true, y_pred, feature_to_predict) 
            logger.info(f"Validation metrics: RMSE={rmse:.2f}, R2={r2:.2f}")
        else:
            logger.info("Prediction complete")
        
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

    finally:
        warnings.filterwarnings('default')