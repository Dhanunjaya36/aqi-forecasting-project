"""
models.py - Train and evaluate all machine learning models

WHAT THIS FILE DOES:
- Trains all 6 machine learning models for comparison
- Evaluates each model using the same metrics
- Returns results table and predictions for all models

THE 6 MODELS:
1. Linear Regression - Simple baseline
2. Ridge Regression - L2 regularization (prevents overfitting)
3. Lasso Regression - L1 regularization (feature selection)
4. Random Forest - Bagging ensemble (many trees, average)
5. Gradient Boosting - Boosting ensemble (sequential error correction)
6. XGBoost - Optimized boosting with regularization

"""

import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import RIDGE_ALPHA, LASSO_ALPHA, RF_PARAMS, GB_PARAMS, XGB_PARAMS

logger = logging.getLogger(__name__)


# =============================================================================
# FUNCTION: evaluate_model
# =============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """
    Train a model, make predictions, and calculate evaluation metrics
    
    This function is used by train_all_models to evaluate each model.
    It returns metrics and predictions for later analysis.
    """
    # Train the model on training data
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))      # Root Mean Square Error
    mae = mean_absolute_error(y_test, y_pred)                # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)                            # R-squared
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 0.001))) * 100  # MAPE
    
    # Store metrics in dictionary
    metrics = {
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    logger.info(f"   {name}: RMSE={rmse:.4f}, R²={r2:.4f}")
    return metrics, y_pred


# =============================================================================
# FUNCTION: train_all_models
# =============================================================================
def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 models and compare performance
    
    This is the main function called from main.py.
    It trains each model, collects results, and returns a comparison table.
    """
    logger.info("="*70)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*70)
    
    # Convert pandas DataFrames to numpy arrays for faster processing
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Handle any remaining NaN values (safety check)
    if np.isnan(X_train_np).any() or np.isnan(X_test_np).any():
        logger.info("   Handling NaN values...")
        imputer = SimpleImputer(strategy='mean')
        X_train_np = imputer.fit_transform(X_train_np)
        X_test_np = imputer.transform(X_test_np)
    
    results = []      # Store metrics for each model
    predictions = {}  # Store predictions for each model

    
    
    # ----- 1. LINEAR REGRESSION -----
    # Baseline model - assumes linear relationship
    logger.info("\n Linear Regression...")
    metrics, pred = evaluate_model(LinearRegression(), X_train_np, y_train_np, X_test_np, y_test_np, "Linear")
    results.append(metrics)
    predictions['Linear'] = pred

    
    # ----- 2. RIDGE REGRESSION -----
    # L2 regularization - adds penalty to large coefficients
    logger.info("\n Ridge Regression...")
    metrics, pred = evaluate_model(Ridge(alpha=RIDGE_ALPHA), X_train_np, y_train_np, X_test_np, y_test_np, "Ridge")
    results.append(metrics)
    predictions['Ridge'] = pred

    
    # ----- 3. LASSO REGRESSION -----
    # L1 regularization - can make coefficients zero (feature selection)
    logger.info("\n Lasso Regression...")
    metrics, pred = evaluate_model(Lasso(alpha=LASSO_ALPHA, max_iter=10000), X_train_np, y_train_np, X_test_np, y_test_np, "Lasso")
    results.append(metrics)
    predictions['Lasso'] = pred
    
    # ----- 4. RANDOM FOREST -----
    # Bagging ensemble - many trees, average their predictions
    logger.info("\n Random Forest...")
    metrics, pred = evaluate_model(RandomForestRegressor(**RF_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "Random Forest")
    results.append(metrics)
    predictions['Random Forest'] = pred


    
    # ----- 5. GRADIENT BOOSTING -----
    # Boosting ensemble - sequential error correction
    logger.info("\n Gradient Boosting...")
    metrics, pred = evaluate_model(GradientBoostingRegressor(**GB_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "Gradient Boosting")
    results.append(metrics)
    predictions['Gradient Boosting'] = pred



    
    # ----- 6. XGBOOST -----
    # Optimized boosting - regularization + parallel processing
    try:
        import xgboost as xgb
        logger.info("\n XGBoost...")
        metrics, pred = evaluate_model(xgb.XGBRegressor(**XGB_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "XGBoost")
        results.append(metrics)
        predictions['XGBoost'] = pred
    except ImportError:
        logger.warning("XGBoost not installed - skipping")
    


    # Create DataFrame with all results, sorted by RMSE (best first)
    results_df = pd.DataFrame(results).sort_values('RMSE')
    
    # Print best model
    best = results_df.iloc[0]
    logger.info(f"\ BEST MODEL: {best['Model']}")
    logger.info(f"   RMSE: {best['RMSE']:.4f}")
    logger.info(f"   R²:   {best['R2']:.4f}")
    
    return results_df, predictions
