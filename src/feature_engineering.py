"""
feature_engineering.py - Create features for time series forecasting
"""

import pandas as pd
import numpy as np
import logging
from .config import LAG_HOURS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

def find_target_column(df):
    """
    Find the CO column automatically
    
    Args:
        df: DataFrame
        
    Returns:
        str: Name of the target column
    """
    for col in df.columns:
        if 'CO' in col.upper() and 'lag' not in col.lower() and 'diff' not in col.lower():
            return col
    return df.columns[0]

def create_features(df):
    """
    Create features for time series forecasting
    
    Features created:
    1. Temporal features (hour, day, month, weekend)
    2. Cyclical encoding (sin/cos for circular time)
    3. Lag features (past values)
    4. Rolling statistics (trends and volatility)
    5. Interaction features (temperature-humidity)
    
    Args:
        df: Model-ready DataFrame
        
    Returns:
        tuple: (feature_dataframe, target_column_name)
    """
    logger.info("Creating features...")
    
    df_feat = df.copy()
    target_col = find_target_column(df_feat)
    logger.info(f"   Target: {target_col}")
    logger.info(f"   Target range: {df_feat[target_col].min():.2f} to {df_feat[target_col].max():.2f}")
    
    # 1. Temporal features
    df_feat['hour'] = df_feat.index.hour
    df_feat['day'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    
    # 2. Cyclical encoding (handles circular nature of time)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    # 3. Lag features (CRITICAL for time series!)
    logger.info("   Creating lag features...")
    for lag in LAG_HOURS:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
        if lag in [1, 6, 12, 24]:
            logger.info(f"      Created lag_{lag}")
    
    # 4. Rolling statistics
    logger.info("   Creating rolling statistics...")
    for window in ROLLING_WINDOWS:
        df_feat[f'{target_col}_rolling_mean_{window}'] = df_feat[target_col].rolling(window).mean()
        df_feat[f'{target_col}_rolling_std_{window}'] = df_feat[target_col].rolling(window).std()
        logger.info(f"      Created rolling_{window}h")
    
    # 5. Other pollutant lag features
    if 'NOx_GT' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'NOx_lag_{lag}'] = df_feat['NOx_GT'].shift(lag)
        logger.info("   Created NOx lag features")
    
    if 'T' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'T_lag_{lag}'] = df_feat['T'].shift(lag)
        logger.info("   Created Temperature lag features")
    
    if 'RH' in df_feat.columns:
        for lag in [1, 6, 24]:
            df_feat[f'RH_lag_{lag}'] = df_feat['RH'].shift(lag)
        logger.info("   Created Humidity lag features")
    
    # 6. Interaction features
    if 'T' in df_feat.columns and 'RH' in df_feat.columns:
        df_feat['temp_humidity'] = df_feat['T'] * df_feat['RH'] / 100
        logger.info("   Created temp_humidity interaction")
    
    # 7. Remove first 24 rows (where lag features have NaN)
    initial = len(df_feat)
    df_feat = df_feat.iloc[24:].copy()
    logger.info(f"   Removed first 24 rows: {initial} → {len(df_feat)}")
    
    # 8. Fill any remaining NaN values with column mean
    logger.info("   Filling remaining NaN...")
    for col in df_feat.columns:
        if df_feat[col].isnull().any():
            df_feat[col] = df_feat[col].fillna(df_feat[col].mean())
    
    logger.info(f"   Final shape: {df_feat.shape}")
    
    return df_feat, target_col
