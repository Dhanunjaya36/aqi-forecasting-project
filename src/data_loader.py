"""
data_loader.py - Load raw dataset from UCI repository
"""

import pandas as pd
import logging
from .config import RAW_DATA_FILE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data():
    """
    Load raw AQI dataset from CSV file
    
    Returns:
        DataFrame: Raw data with 9,447 rows × 17 columns
    """
    logger.info("Loading raw data...")
    
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"File not found: {RAW_DATA_FILE}")
    
    # Read CSV with correct parameters:
    # sep=';' because UCI uses semicolon separator
    # decimal=',' because UCI uses comma for decimals
    df = pd.read_csv(RAW_DATA_FILE, sep=';', decimal=',')
    
    logger.info(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df
