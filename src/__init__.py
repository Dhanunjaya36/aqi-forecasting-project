"""
AQI Forecasting Project - Source Code Package
MSc Data Science - University of Hertfordshire

This package contains all the modules for the AQI forecasting project.

WHAT THIS FILE DOES:
- Makes the 'src' folder a Python package
- Exports all important functions so they can be imported easily
- Acts as a central hub for the entire project

HOW IT'S USED:
    In main.py, we write: from src import load_raw_data, clean_data, ...
    Instead of: from src.data_loader import load_raw_data

Author: Dhanunjaya Rao Thandra
Date: April 2026
"""


# =============================================================================
# PACKAGE METADATA
# =============================================================================
__version__ = "1.0.0"
__author__ = "Dhanunjaya Rao Thandra"


# =============================================================================
# IMPORT FROM CONFIG MODULE
# =============================================================================
from .config import (
    BASE_PATH, RAW_DATA_PATH, CLEANED_DATA_PATH, RESULTS_PATH, PLOTS_PATH,
    RAW_DATA_FILE, CLEANED_DATA_FILE, MODEL_READY_FILE, FEATURES_FILE,
    TEST_SIZE, RANDOM_STATE, LAG_HOURS, ROLLING_WINDOWS,
    XGB_PARAMS, RF_PARAMS, GB_PARAMS, RIDGE_ALPHA, LASSO_ALPHA
)


# =============================================================================
# IMPORT FROM DATA LOADER MODULE
# =============================================================================
from .data_loader import load_raw_data


# =============================================================================
# IMPORT FROM PREPROCESSOR MODULE
# =============================================================================
from .preprocessor import (
    clean_data,
    save_cleaned_data,
    create_model_ready
)


# =============================================================================
# IMPORT FROM FEATURE ENGINEERING MODULE
# =============================================================================
from .feature_engineering import (
    create_features,
    find_target_column
)


# =============================================================================
# IMPORT FROM MODELS MODULE
# =============================================================================
from .models import (
    train_all_models,
    evaluate_model
)


# =============================================================================
# IMPORT FROM EVALUATOR MODULE
# =============================================================================
from .evaluator import (
    calculate_metrics,
    compare_xgboost_tuning
)


# =============================================================================
# IMPORT FROM VISUALIZER MODULE
# =============================================================================
from .visualizer import (
    plot_model_comparison,
    plot_predictions,
    plot_error_distribution,
    plot_hyperparameter_comparison,
    plot_residual_analysis,
    plot_feature_importance,
    plot_full_dashboard,
    plot_residual_vs_predicted  
)


# =============================================================================
# PACKAGE LOADING CONFIRMATION
# =============================================================================
print(f"AQI Forecasting Package v{__version__} loaded")
print(f"   Author: {__author__}")
print(f"   Available functions: load_raw_data, clean_data, create_features, "
      f"train_all_models, plot_model_comparison, plot_predictions, "
      f"plot_error_distribution, plot_hyperparameter_comparison, "
      f"plot_residual_analysis, plot_feature_importance, plot_full_dashboard, "
      f"plot_residual_vs_predicted")
