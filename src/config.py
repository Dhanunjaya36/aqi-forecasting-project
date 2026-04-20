"""
config.py - Configuration settings for the project
All paths and parameters are defined here


PURPOSE OF THIS FILE:
- Centralizes all settings in one place
- Makes it easy to change parameters without hunting through code
- Ensures consistency across all modules



WHAT IT CONTAINS:
- File paths (where data is stored)
- Model parameters (test size, random state)
- Feature engineering settings (lag hours, rolling windows)
- Hyperparameters for all models

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import os
from pathlib import Path



# =============================================================================
# PROJECT PATHS
# =============================================================================
# These paths tell the program where to find and save files

# Get the project root directory
# __file__ is the path to this file (src/config.py)
# .parent goes up one level (to src/)
# .parent again goes up to the project root (AQI_Project/)
PROJECT_ROOT = Path(__file__).parent.parent

# BASE_PATH is the same as PROJECT_ROOT - the main project folder
BASE_PATH = PROJECT_ROOT

# Data paths - where datasets are stored
DATA_DIR = BASE_PATH / "data"                      # Main data folder
RAW_DATA_PATH = DATA_DIR / "raw data"              # Raw dataset location
CLEANED_DATA_PATH = DATA_DIR / "cleaned data"      # Cleaned data location

# Results paths - where outputs are saved
RESULTS_PATH = BASE_PATH / "results"               # Main results folder
PLOTS_PATH = RESULTS_PATH / "plots"                # Plots subfolder

# Create all directories if they don't exist
# This ensures the program doesn't crash if folders are missing
for path in [RAW_DATA_PATH, CLEANED_DATA_PATH, RESULTS_PATH, PLOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)        # exist_ok=True means don't error if exists



# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Full file paths for specific files
# These combine the folder paths with filenames

RAW_DATA_FILE = RAW_DATA_PATH / "AirQualityUCI.csv"           # Original UCI dataset
CLEANED_DATA_FILE = CLEANED_DATA_PATH / "air_quality_cleaned.csv"     # Cleaned data
MODEL_READY_FILE = CLEANED_DATA_PATH / "air_quality_cleaned_model_ready.csv"  # With cleaned column names
FEATURES_FILE = CLEANED_DATA_PATH / "air_quality_features.csv"            # With all features



# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# These control how models are trained

# Train-test split: 80% for training, 20% for testing
TEST_SIZE = 0.2
# Random state ensures the same random numbers each run (reproducibility)
RANDOM_STATE = 42

# Time series parameters
SEQ_LENGTH = 24      # Sequence length for lag features (look back 24 hours)

# Feature engineering parameters - these control what features are created
LAG_HOURS = [1, 3, 6, 12, 24]          # Hours to create lag features for
ROLLING_WINDOWS = [3, 6, 12, 24]       # Window sizes for rolling statistics



# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# These are the tuned parameters that gave the best performance
# Each model has its own set of parameters

# XGBoost parameters (optimized after tuning)
# - n_estimators: 300 trees (default is 100)
# - max_depth: 8 layers deep (default is 6) - captures complex patterns
# - learning_rate: 0.05 (default is 0.3) - slower learning for better convergence
# - subsample: 0.8 - uses 80% of data per tree (prevents overfitting)
# - colsample_bytree: 0.8 - uses 80% of features per tree (prevents overfitting)
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

# Random Forest parameters
# - n_estimators: 200 trees
# - max_depth: 15 layers deep
# - n_jobs: -1 means use all CPU cores (faster training)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Gradient Boosting parameters
# - n_estimators: 200 trees
# - max_depth: 5 layers deep (shallower than XGBoost)
# - learning_rate: 0.05
GB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'random_state': RANDOM_STATE
}

# Ridge and Lasso parameters
# - alpha controls regularization strength
# - Higher alpha = more regularization (simpler model)
RIDGE_ALPHA = 1.0          # Ridge uses L2 regularization
LASSO_ALPHA = 0.001        # Lasso uses L1 regularization (can make coefficients zero)



# =============================================================================
# CONFIGURATION TEST (when run directly)
# =============================================================================
# This block runs only if this file is executed directly (not imported)
# It prints the configuration to verify everything is set up correctly
if __name__ == "__main__":
    print("="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(" Configuration loaded successfully!")
    print(f"\n PROJECT PATHS:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Raw Data Path: {RAW_DATA_PATH}")
    print(f"   Cleaned Data Path: {CLEANED_DATA_PATH}")
    print(f"   Results Path: {RESULTS_PATH}")
    print(f"   Plots Path: {PLOTS_PATH}")
    
    print(f"\n MODEL PARAMETERS:")
    print(f"   Test Size: {TEST_SIZE*100:.0f}%")
    print(f"   Random State: {RANDOM_STATE}")
    print(f"   Sequence Length: {SEQ_LENGTH} hours")
    
    print(f"\n FEATURE ENGINEERING:")
    print(f"   Lag Hours: {LAG_HOURS}")
    print(f"   Rolling Windows: {ROLLING_WINDOWS}")
    
    print(f"\n XGBOOST PARAMETERS:")
    for key, value in XGB_PARAMS.items():
        print(f"   {key}: {value}")
    
    print(f"\n All configurations ready for use!")
