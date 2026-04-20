"""
preprocessor.py - Data cleaning and preprocessing functions
"""

import pandas as pd
import numpy as np
import logging
from .config import CLEANED_DATA_PATH, CLEANED_DATA_FILE, MODEL_READY_FILE

logger = logging.getLogger(__name__)

def clean_data(df_raw):
    """
    Clean the raw AQI dataset
    
    Handles:
    - -200 values (missing data)
    - Impossible temperatures (-10°C to 45°C)
    - Negative pollutant values
    - Creates datetime index
    
    Args:
        df_raw: Raw DataFrame
        
    Returns:
        DataFrame: Cleaned data
    """
    logger.info("Cleaning data...")
    
    # Create datetime column
    df_raw['DateTime'] = pd.to_datetime(
        df_raw['Date'] + ' ' + df_raw['Time'],
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )
    
    # Replace -200 with NaN (missing values)
    df_raw = df_raw.replace(-200, np.nan)
    
    # Fix temperature (Rome realistic range: -10°C to 45°C)
    df_raw.loc[(df_raw['T'] < -10) | (df_raw['T'] > 45), 'T'] = np.nan
    
    # Fix humidity (must be between 0% and 100%)
    df_raw.loc[(df_raw['RH'] < 0) | (df_raw['RH'] > 100), 'RH'] = np.nan
    
    # Fix negative pollutant values
    df_raw.loc[df_raw['CO(GT)'] < 0, 'CO(GT)'] = np.nan
    
    # Fill missing values - forward fill then backward fill for time series
    df_raw = df_raw.ffill().bfill()
    
    # Set DateTime as index and remove original Date/Time columns
    df_raw.set_index('DateTime', inplace=True)
    df_raw.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')
    
    logger.info(f"Cleaned: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    return df_raw

def save_cleaned_data(df, filename):
    """
    Save cleaned data to CSV
    
    Args:
        df: DataFrame to save
        filename: Name of the file
    """
    filepath = CLEANED_DATA_PATH / filename
    df.to_csv(filepath)
    logger.info(f"Saved: {filepath}")

def create_model_ready(df):
    """
    Create version with cleaned column names for modeling
    
    Removes special characters like () and . from column names
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame: Model-ready data with cleaned column names
    """
    df_model = df.copy()
    df_model.columns = df_model.columns.str.replace(r'[()]', '', regex=True).str.replace('.', '_', regex=False)
    return df_model
