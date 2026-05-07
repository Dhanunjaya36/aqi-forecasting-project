"""
feature_engineering.py - Create features for time series forecasting

WHAT THIS FILE DOES:
- Creates temporal features (hour, day, month, weekend)
- Creates cyclical encoding (sin/cos for circular time)
- Creates lag features (past values at different time steps)
- Creates rolling statistics (moving averages and standard deviations)
- Creates interaction features (temperature-humidity combinations)

WHY FEATURE ENGINEERING IS IMPORTANT:
- Raw time series data alone is not enough for good predictions
- Lag features capture temporal dependencies
- Rolling statistics capture trends and volatility
- These features give the model more patterns to learn

Author: Dhanunjaya Rao Thandra
Date: April 2026
"""

import pandas as pd
import numpy as np
import logging
from .config import LAG_HOURS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def find_target_column(df):
    """
    Find the CO column automatically
    
    This function searches through all column names to find the one
    that contains "CO" in its name. This makes the code adaptable
    to different column naming conventions.
    
    Args:
        df: DataFrame with the data
        
    Returns:
        str: Name of the target column (the CO column)
    """
    # Loop through all column names
    for col in df.columns:
        # Check if column contains "CO" (case insensitive)
        # Also exclude columns that already have 'lag' or 'diff' in name
        # These are engineered features, not the original target
        if 'CO' in col.upper() and 'lag' not in col.lower() and 'diff' not in col.lower():
            return col
    
    # If no CO column found, return the first column as fallback
    return df.columns[0]


def create_features(df):
    """
    Create features for time series forecasting
    
    This function adds multiple types of features to the dataset:
    1. Temporal features (hour, day, month, weekend)
    2. Cyclical encoding (sin/cos for circular time)
    3. Lag features (past values)
    4. Rolling statistics (trends and volatility)
    5. Interaction features (combined effects)
    
    Args:
        df: Model-ready DataFrame with datetime index
        
    Returns:
        tuple: (feature_dataframe, target_column_name)
    """
    
    logger.info("Creating features...")
    
    # Create a copy to avoid modifying the original
    df_feat = df.copy()
    
    # Find which column is our target (CO)
    target_col = find_target_column(df_feat)
    logger.info(f"   Target: {target_col}")
    logger.info(f"   Target range: {df_feat[target_col].min():.2f} to {df_feat[target_col].max():.2f}")
    
    # =========================================================================
    # 1. TEMPORAL FEATURES
    # =========================================================================
    # Extract time components from the datetime index
    df_feat['hour'] = df_feat.index.hour                     # Hour of day (0-23)
    df_feat['day'] = df_feat.index.day                       # Day of month (1-31)
    df_feat['month'] = df_feat.index.month                   # Month (1-12)
    df_feat['dayofweek'] = df_feat.index.dayofweek           # Day of week (0=Monday, 6=Sunday)
    df_feat['weekend'] = (df_feat['dayofweek'] >= 5).astype(int)   # 1 if weekend, else 0
    
    # =========================================================================
    # 2. CYCLICAL ENCODING
    # =========================================================================
    # Why cyclical encoding?
    # Without it, a model would think hour 23 is far from hour 0
    # But they are only 1 hour apart! Sin/cos solves this by placing time on a circle
    
    # Hour of day on a circle (24-hour cycle)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    
    # Month of year on a circle (12-month cycle)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    # =========================================================================
    # 3. LAG FEATURES
    # =========================================================================
    # Lag features answer: "What was the CO value X hours ago?"
    # Critical for time series because past values predict future values
    logger.info("   Creating lag features...")
    
    for lag in LAG_HOURS:
        # Shift the target column by 'lag' hours to create lag feature
        # Example: For lag=1, each row gets the CO value from 1 hour ago
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
        
        # Log progress for key lag values
        if lag in [1, 6, 12, 24]:
            logger.info(f"      Created lag_{lag}")
    
    # =========================================================================
    # 4. ROLLING STATISTICS
    # =========================================================================
    # Rolling statistics answer: "What is the trend over the last X hours?"
    # Rolling mean smooths out noise, rolling std measures volatility
    logger.info("   Creating rolling statistics...")
    
    for window in ROLLING_WINDOWS:
        # Rolling mean (moving average) - captures trend
        # Example: For window=3, average of current and 2 previous values
        df_feat[f'{target_col}_rolling_mean_{window}'] = df_feat[target_col].rolling(window).mean()
        
        # Rolling standard deviation - measures volatility
        # High standard deviation means unstable conditions
        df_feat[f'{target_col}_rolling_std_{window}'] = df_feat[target_col].rolling(window).std()
        
        logger.info(f"      Created rolling_{window}h")
    
    # =========================================================================
    # 5. OTHER POLLUTANT LAG FEATURES
    # =========================================================================
    # Create lag features for other important variables
    # Pollution doesn't depend only on CO - weather and other pollutants matter
    
    # NOx lag features (Nitrogen Oxides)
    if 'NOx_GT' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'NOx_lag_{lag}'] = df_feat['NOx_GT'].shift(lag)
        logger.info("   Created NOx lag features")
    
    # Temperature lag features
    if 'T' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'T_lag_{lag}'] = df_feat['T'].shift(lag)
        logger.info("   Created Temperature lag features")
    
    # Humidity lag features
    if 'RH' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'RH_lag_{lag}'] = df_feat['RH'].shift(lag)
        logger.info("   Created Humidity lag features")
    
    # =========================================================================
    # 6. INTERACTION FEATURES
    # =========================================================================
    # Interaction features capture combined effects
    # Example: Temperature and humidity together affect pollution dispersion
    if 'T' in df_feat.columns and 'RH' in df_feat.columns:
        df_feat['temp_humidity'] = df_feat['T'] * df_feat['RH'] / 100
        logger.info("   Created temp_humidity interaction")
    
    # =========================================================================
    # 7. HANDLE NaN VALUES - REMOVE FIRST 24 ROWS
    # =========================================================================
    # Lag features create NaN for the first few rows
    # For lag_24, the first 24 rows will have NaN (no previous data)
    # Remove these rows since they can't be used for training
    
    initial = len(df_feat)                              # Saves original number of rows
    df_feat = df_feat.iloc[24:].copy()                  # Takes rows from position 24 to the end (skips first 24)
    logger.info(f"   Removed first 24 rows: {initial} → {len(df_feat)}")
    
    # =========================================================================
    # 8. FIX: REMOVE COLUMNS THAT ARE ALL NaN VALUES
    # =========================================================================
    # After removing the first 24 rows, some columns may become completely empty
    # This happens if a column only had values in the first 24 rows
    # These columns cannot be used for training, so we remove them
    
    columns_before = len(df_feat.columns)
    df_feat = df_feat.dropna(axis=1, how='all')         # Drop columns where ALL values are NaN
    columns_removed = columns_before - len(df_feat.columns)
    
    if columns_removed > 0:
        logger.info(f"   Removed {columns_removed} columns that had all NaN values")
    
    # =========================================================================
    # 9. FILL REMAINING NaN VALUES WITH COLUMN MEAN
    # =========================================================================
    # There may still be random NaN values from rolling statistics
    # Fill them with the column mean (simple imputation)
    # This preserves the overall distribution without introducing bias
    
    logger.info("   Filling remaining NaN...")
    
    for col in df_feat.columns:
        # Check if column has ANY NaN values
        if df_feat[col].isnull().any():
            # Replace each NaN with column average
            # Example: Column [2.5, NaN, 3.1, 2.8] with mean=2.8 becomes [2.5, 2.8, 3.1, 2.8]
            df_feat[col] = df_feat[col].fillna(df_feat[col].mean())
    
    logger.info(f"   Final shape: {df_feat.shape}")
    
    return df_feat, target_col
