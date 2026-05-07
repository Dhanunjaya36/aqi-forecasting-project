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

Author: Dhanunjaya Rao Thandra
Date: April 2026
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
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
    
    Args:
        model: The machine learning model to train
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        name: Name of the model for display
        
    Returns:
        tuple: (metrics_dict, predictions_array)
    """
    
    # Train the model on training data
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    # RMSE penalizes large errors more than small ones
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # MAE is the average absolute error in original units
    mae = mean_absolute_error(y_test, y_pred)
    
    # R² measures how much variance the model explains (1.0 is perfect)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE is the error as a percentage (easier to understand)
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 0.001))) * 100
    
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
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target values
        y_test: Test target values
        
    Returns:
        tuple: (results_dataframe, predictions_dictionary)
    """
    
    logger.info("="*70)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*70)
    
    # Convert pandas DataFrames to numpy arrays for faster processing
    # scikit-learn works faster with numpy than pandas
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
        X_test_np = X_test.values
        y_train_np = y_train.values
        y_test_np = y_test.values
    else:
        X_train_np = X_train
        X_test_np = X_test
        y_train_np = y_train
        y_test_np = y_test
    
    # =========================================================================
    # FIX 1: Remove columns that are ALL NaN values
    # This fixes the warning: "Skipping features without any observed values"
    # =========================================================================
    # Find which columns have at least one non-NaN value
    valid_columns = ~np.isnan(X_train_np).all(axis=0)
    
    # If there are columns with all NaN values, remove them
    if not valid_columns.all():
        removed_count = np.sum(~valid_columns)
        X_train_np = X_train_np[:, valid_columns]
        X_test_np = X_test_np[:, valid_columns]
        logger.info(f"   Removed {removed_count} columns that had all NaN values")
    
    # =========================================================================
    # FIX 2: Handle any remaining NaN values with mean imputation
    # This is a safety net for any random NaN values that might remain
    # =========================================================================
    if np.isnan(X_train_np).any() or np.isnan(X_test_np).any():
        logger.info("   Handling remaining NaN values with mean imputation...")
        imputer = SimpleImputer(strategy='mean')
        X_train_np = imputer.fit_transform(X_train_np)
        X_test_np = imputer.transform(X_test_np)
        logger.info("   NaN values imputed successfully")
    
    # Store results and predictions
    results = []
    predictions = {}
    
    # ----- MODEL 1: LINEAR REGRESSION -----
    # Baseline model - assumes linear relationship between features and target
    # Formula: CO = w1*x1 + w2*x2 + ... + bias
    logger.info("\n Linear Regression...")
    metrics, pred = evaluate_model(LinearRegression(), X_train_np, y_train_np, X_test_np, y_test_np, "Linear")
    results.append(metrics)
    predictions['Linear'] = pred
    
    # ----- MODEL 2: RIDGE REGRESSION -----
    # L2 regularization - adds penalty to large coefficients
    # Prevents overfitting by shrinking coefficients
    # Alpha controls the strength of regularization
    logger.info("\n Ridge Regression...")
    metrics, pred = evaluate_model(Ridge(alpha=RIDGE_ALPHA), X_train_np, y_train_np, X_test_np, y_test_np, "Ridge")
    results.append(metrics)
    predictions['Ridge'] = pred
    
    # ----- MODEL 3: LASSO REGRESSION -----
    # L1 regularization - can make coefficients exactly zero
    # Performs automatic feature selection
    # Alpha=0.001 gives gentle regularization
    logger.info("\n Lasso Regression...")
    metrics, pred = evaluate_model(Lasso(alpha=LASSO_ALPHA, max_iter=10000), X_train_np, y_train_np, X_test_np, y_test_np, "Lasso")
    results.append(metrics)
    predictions['Lasso'] = pred
    
    # ----- MODEL 4: RANDOM FOREST -----
    # Bagging ensemble - creates many decision trees and averages predictions
    # Each tree sees a random subset of data and features
    # Reduces overfitting compared to single decision tree
    logger.info("\n Random Forest...")
    metrics, pred = evaluate_model(RandomForestRegressor(**RF_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "Random Forest")
    results.append(metrics)
    predictions['Random Forest'] = pred
    
    # ----- MODEL 5: GRADIENT BOOSTING -----
    # Boosting ensemble - builds trees sequentially
    # Each new tree tries to correct errors of previous trees
    # Different from Random Forest which builds trees in parallel
    logger.info("\n Gradient Boosting...")
    metrics, pred = evaluate_model(GradientBoostingRegressor(**GB_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "Gradient Boosting")
    results.append(metrics)
    predictions['Gradient Boosting'] = pred
    
    # ----- MODEL 6: XGBOOST -----
    # Optimized gradient boosting with additional features:
    # - Regularization (L1 and L2) to prevent overfitting
    # - Parallel processing for faster training
    # - Built-in handling of missing values
    logger.info("\n XGBoost...")
    try:
        import xgboost as xgb
        metrics, pred = evaluate_model(xgb.XGBRegressor(**XGB_PARAMS), X_train_np, y_train_np, X_test_np, y_test_np, "XGBoost")
        results.append(metrics)
        predictions['XGBoost'] = pred
    except ImportError:
        logger.warning("XGBoost not installed - skipping this model")
        logger.warning("To install XGBoost, run: pip install xgboost")
    
    # Create DataFrame with all results, sorted by RMSE (lowest is best)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    # Print best model
    best = results_df.iloc[0]
    logger.info(f"\n BEST MODEL: {best['Model']}")
    logger.info(f"   RMSE: {best['RMSE']:.4f} mg/m³")
    logger.info(f"   R²:   {best['R2']:.4f}")
    
    return results_df, predictions
