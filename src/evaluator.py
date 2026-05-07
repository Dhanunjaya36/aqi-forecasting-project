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

Author: Dhanunjaya Rao Thandra
Date: April 2026
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
       - Formula: square root of (sum of (actual - predicted) squared divided by n)
       - Lower is better
    
    2. MAE (Mean Absolute Error):
       - Average of absolute differences
       - Treats all errors equally
       - Formula: sum of absolute(actual - predicted) divided by n
       - Lower is better
    
    3. R² (R-squared / Coefficient of Determination):
       - How much variance the model explains
       - 1.0 = perfect, 0.0 = no better than guessing average
       - Formula: 1 - (sum of squared errors divided by total sum of squares)
       - Higher is better
    
    4. MAPE (Mean Absolute Percentage Error):
       - Error as a percentage
       - Easy for non-technical people to understand
       - Formula: (100 divided by n) times sum of absolute((actual - predicted)/actual)
       - Lower is better
    
    Args:
        y_true: Actual values (what really happened)
        y_pred: Predicted values (what the model said)
        model_name: Name of the model (for display)
        
    Returns:
        dict: Dictionary containing all four metrics
    """
    
    # Calculate RMSE - penalizes large errors
    # If one prediction is way off, RMSE increases significantly
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAE - simple average error in original units
    # If MAE = 0.17, predictions are off by 0.17 units on average
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R² - how much variance is explained
    # R² = 0.94 means model explains 94% of variation
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE - error as percentage
    # Add small epsilon (0.001) to avoid division by zero
    # MAPE = 15% means predictions are 15% off on average
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
        X_train: Training features
        X_test: Test features
        y_train: Training target values
        y_test: Test target values
        
    Returns:
        tuple: (default_metrics, tuned_metrics, predictions_tuple)
               Returns (None, None, None) if XGBoost not installed
    """
    
    # Try to import XGBoost - it might not be installed
    try:
        import xgboost as xgb
        
        # Convert to numpy arrays for faster processing
        # Handle both pandas DataFrames and numpy arrays
        # .values converts pandas DataFrame to numpy array
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
        
        # Default XGBoost (before tuning)
        # Uses sklearn's default parameters:
        # - n_estimators: 100 trees
        # - max_depth: 6 levels
        # - learning_rate: 0.3
        # - No subsampling (uses all data)
        print("\n" + "="*50)
        print("DEFAULT XGBoost (No Tuning)")
        print("="*50)
        
        # Create and train default model
        xgb_default = xgb.XGBRegressor(random_state=42)
        xgb_default.fit(X_train_np, y_train_np)
        y_pred_default = xgb_default.predict(X_test_np)
        
        # Calculate metrics for default version
        default_metrics = calculate_metrics(y_test_np, y_pred_default, "Default XGBoost")
        
        # Print default results
        print("\nDefault Parameters:")
        print("  - n_estimators: 100")
        print("  - max_depth: 6")
        print("  - learning_rate: 0.3")
        print("  - subsample: 1.0")
        print("  - colsample_bytree: 1.0")
        
        print("\nPerformance:")
        print(f"  - RMSE: {default_metrics['RMSE']:.4f} mg/m^3")
        print(f"  - MAE: {default_metrics['MAE']:.4f} mg/m^3")
        print(f"  - R²: {default_metrics['R2']:.4f}")
        print(f"  - MAPE: {default_metrics['MAPE']:.2f}%")
        
        # Tuned XGBoost (after tuning)
        # Optimized parameters after manual testing:
        # - n_estimators: 300 trees (more trees = better learning)
        # - max_depth: 8 layers (deeper trees = capture complex patterns)
        # - learning_rate: 0.05 (slower learning = better convergence)
        # - subsample: 0.8 (use 80% of data per tree = prevent overfitting)
        # - colsample_bytree: 0.8 (use 80% of features per tree = prevent overfitting)
        print("\n" + "="*50)
        print("Tuned XGBoost (After Tuning)")
        print("="*50)
        
        # Create and train tuned model
        xgb_tuned = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_tuned.fit(X_train_np, y_train_np)
        y_pred_tuned = xgb_tuned.predict(X_test_np)
        
        # Calculate metrics for tuned version
        tuned_metrics = calculate_metrics(y_test_np, y_pred_tuned, "Tuned XGBoost")
        
        # Print tuned results
        print("\nTuned Parameters:")
        print("  - n_estimators: 300")
        print("  - max_depth: 8")
        print("  - learning_rate: 0.05")
        print("  - subsample: 0.8")
        print("  - colsample_bytree: 0.8")
        
        print("\nPerformance:")
        print(f"  - RMSE: {tuned_metrics['RMSE']:.4f} mg/m^3")
        print(f"  - MAE: {tuned_metrics['MAE']:.4f} mg/m^3")
        print(f"  - R²: {tuned_metrics['R2']:.4f}")
        print(f"  - MAPE: {tuned_metrics['MAPE']:.2f}%")
        
        # Calculate and print improvement
        improvement_rmse = ((default_metrics['RMSE'] - tuned_metrics['RMSE']) / default_metrics['RMSE']) * 100
        improvement_r2 = ((tuned_metrics['R2'] - default_metrics['R2']) / default_metrics['R2']) * 100
        improvement_mae = ((default_metrics['MAE'] - tuned_metrics['MAE']) / default_metrics['MAE']) * 100
        
        print("\n" + "="*50)
        print("Hyperparameter Tuning Improvement")
        print("="*50)
        print(f"\n  - RMSE decreased by: {improvement_rmse:.1f}%")
        print(f"    (from {default_metrics['RMSE']:.4f} to {tuned_metrics['RMSE']:.4f})")
        print(f"  - MAE decreased by: {improvement_mae:.1f}%")
        print(f"    (from {default_metrics['MAE']:.4f} to {tuned_metrics['MAE']:.4f})")
        print(f"  - R² increased by: {improvement_r2:.1f}%")
        print(f"    (from {default_metrics['R2']:.4f} to {tuned_metrics['R2']:.4f})")
        
        # Return all results for further use (like plotting)
        return default_metrics, tuned_metrics, (y_pred_default, y_pred_tuned)
        
    except ImportError:
        # Handle case where XGBoost is not installed
        print("\nWarning: XGBoost is not installed. Cannot perform tuning comparison.")
        print("To install XGBoost, run: pip install xgboost")
        return None, None, None
        
    except Exception as e:
        # Handle any other errors
        print(f"\nError in hyperparameter comparison: {e}")
        return None, None, None
