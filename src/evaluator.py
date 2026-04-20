"""
evaluator.py - Model evaluation metrics

PURPOSE OF THIS FILE:
- Calculates all performance metrics for models
- Compares default vs tuned XGBoost to show improvement
- Provides standardized metrics across all models

WHAT IT DOES:
- calculate_metrics(): Computes RMSE, MAE, R², MAPE
- compare_xgboost_tuning(): Shows how tuning improved XGBoost

WHY THIS IS IMPORTANT:
- One metric alone doesn't tell the full story
- Multiple metrics give a complete picture of model performance
- Shows that hyperparameter tuning actually improved results

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# FUNCTION: calculate_metrics
# =============================================================================
def calculate_metrics(y_true, y_pred, model_name=""):
    """
    Calculate all evaluation metrics for a model
    
    WHAT THIS FUNCTION DOES:
    Takes actual values and predicted values, calculates four key metrics
    that tell us how good the model is.
    
    THE FOUR METRICS:
    1. RMSE (Root Mean Square Error):
       - Takes square root of average squared error
       - Penalizes large errors more than small ones
       - Formula: √(Σ(yᵢ - ŷᵢ)² / n)
       - Lower is better
    
    2. MAE (Mean Absolute Error):
       - Average of absolute differences
       - Treats all errors equally
       - Formula: Σ|yᵢ - ŷᵢ| / n
       - Lower is better
    
    3. R² (R-squared / Coefficient of Determination):
       - How much variance the model explains
       - 1.0 = perfect, 0.0 = no better than guessing average
       - Formula: 1 - (SS_res / SS_tot)
       - Higher is better
    
    4. MAPE (Mean Absolute Percentage Error):
       - Error as a percentage
       - Easy for non-technical people to understand
       - Formula: (100%/n) × Σ|(yᵢ - ŷᵢ)/yᵢ|
       - Lower is better
    
    Args:
        y_true: Actual values (what really happened)
        y_pred: Predicted values (what the model said)
        model_name: Name of the model (for display)
        
    Returns:
        dict: Dictionary containing all four metrics
    """
    
    # Calculate RMSE - penalizes large errors
    # Example: If one prediction is way off, RMSE increases significantly
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAE - simple average error in original units
    # Example: If MAE = 0.17, predictions are off by 0.17 units on average
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R² - how much variance is explained
    # Example: R² = 0.94 means model explains 94% of variation
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE - error as percentage
    # Add small epsilon (0.001) to avoid division by zero
    # Example: MAPE = 15% means predictions are 15% off on average
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 0.001))) * 100
    
    # Store all metrics in a dictionary
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    return metrics


# =============================================================================
# FUNCTION: compare_xgboost_tuning
# =============================================================================
def compare_xgboost_tuning(X_train, X_test, y_train, y_test):
    """
    Compare default vs tuned XGBoost performance
    
    WHY THIS FUNCTION EXISTS:
    - Shows that hyperparameter tuning actually improves results
    - Provides evidence that optimization was worth the effort
    - Demonstrates understanding of model tuning
    
    WHAT IT COMPARES:
    1. Default XGBoost (100 trees, depth 6, learning rate 0.3)
    2. Tuned XGBoost (300 trees, depth 8, learning rate 0.05, subsampling)
    
    WHAT IT RETURNS:
    - Metrics for default XGBoost
    - Metrics for tuned XGBoost
    - Predictions for both (for plotting)
    
    Args:
        X_train, X_test: Feature data for training and testing
        y_train, y_test: Target values for training and testing
        
    Returns:
        tuple: (default_metrics, tuned_metrics, predictions_tuple)
               Returns (None, None, None) if XGBoost not installed
    """

    
    # Try to import XGBoost - it might not be installed
    try:
        import xgboost as xgb
        
        # ========== CONVERT TO NUMPY ARRAYS ==========
        # Handle both pandas DataFrames and numpy arrays
        # .values converts pandas DataFrame to numpy array
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        
        # ========== DEFAULT XGBOOST (BEFORE TUNING) ==========
        # This uses sklearn's default parameters:
        # - n_estimators: 100 trees
        # - max_depth: 6 levels
        # - learning_rate: 0.3
        # - No subsampling (uses all data)
        print("\n DEFAULT XGBoost (No Tuning)")
        print("-"*40)
        
        xgb_default = xgb.XGBRegressor(random_state=42)
        xgb_default.fit(X_train_np, y_train_np)
        y_pred_default = xgb_default.predict(X_test_np)
        
        # Calculate metrics for default version
        default_metrics = calculate_metrics(y_test_np, y_pred_default, "Default XGBoost")


        
        # ========== TUNED XGBOOST (AFTER TUNING) ==========
        # Optimized parameters after manual testing:
        # - n_estimators: 300 trees (more trees = better learning)
        # - max_depth: 8 layers (deeper trees = capture complex patterns)
        # - learning_rate: 0.05 (slower learning = better convergence)
        # - subsample: 0.8 (use 80% of data per tree = prevent overfitting)
        # - colsample_bytree: 0.8 (use 80% of features per tree = prevent overfitting)
        print("\n Tuned XGBoost (After Tuning)")
        print("-"*40)
        
        xgb_tuned = xgb.XGBRegressor(
            n_estimators=300,      # More trees
            max_depth=8,           # Deeper trees
            learning_rate=0.05,    # Slower learning
            subsample=0.8,         # 80% data per tree
            colsample_bytree=0.8,  # 80% features per tree
            random_state=42
        )
        xgb_tuned.fit(X_train_np, y_train_np)
        y_pred_tuned = xgb_tuned.predict(X_test_np)
        
        # Calculate metrics for tuned version
        tuned_metrics = calculate_metrics(y_test_np, y_pred_tuned, "Tuned XGBoost")


        
        # ========== CALCULATE IMPROVEMENT ==========
        # Show how much better tuning made the model
        improvement_rmse = ((default_metrics['RMSE'] - tuned_metrics['RMSE']) / default_metrics['RMSE']) * 100
        improvement_r2 = ((tuned_metrics['R2'] - default_metrics['R2']) / default_metrics['R2']) * 100
        
        # Print improvement summary
        print(f"\n HYPERPARAMETER TUNING IMPROVEMENT:")
        print(f"   RMSE decreased by {improvement_rmse:.1f}%")
        print(f"   R² increased by {improvement_r2:.1f}%")
        
        # Return all results
        return default_metrics, tuned_metrics, (y_pred_default, y_pred_tuned)
        
    except Exception as e:
        # Handle case where XGBoost is not installed or other errors
        print(f"Could not run hyperparameter comparison: {e}")
        return None, None, None

