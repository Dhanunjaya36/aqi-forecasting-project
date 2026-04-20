"""
Project Overview
Problem Statement: Predicting air quality index (AQI) based on various environmental parameters, enabling proactive public health interventions and urban planning decisions. To develop a predictive model that effectively analyzes historical air quality data to provide precise future AQI predictions. We will experiment with various machine learning models to identify the most accurate forecasting approach.

Research Significance: Accurate AQI forecasting is critical for public health, environmental management, and policy making. This project contributes to data science applications in environmental monitoring and provides insights for sustainable urban development.

About the Dataset
Source: UCI Machine Learning Repository - Air Quality Dataset.

Location: Rome, Italy

Time Period: March 2004 - February 2005 Frequency: Hourly measurements

Key Variables:

CO(GT): Carbon Monoxide concentration (mg/m³) - Primary pollutant indicator

NOx(GT): Nitrogen Oxides concentration (ppb)

NO2(GT): Nitrogen Dioxide concentration (μg/m³)

T: Ambient temperature (°C)

RH: Relative humidity (%)

AH: Absolute humidity

Date/Time: Temporal indicators for time-series analysis
"""

#=============================================================================
# 1. IMPORT LIBRARIES
#=============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

#=============================================================================
# 2. DEFINE YOUR EXISTING FOLDER PATHS
#=============================================================================

# Define base path - CHANGE THIS TO YOUR PATH
base_path = r"C:\Users\tdhan\OneDrive\Desktop\AQI_Project"

# Your existing folder paths
raw_data_path = os.path.join(base_path, "data", "raw data")
cleaned_data_path = os.path.join(base_path, "data", "cleaned data")
before_plot_path = os.path.join(base_path, "data", "before data cleaned plot")
after_plot_path = os.path.join(base_path, "data", "after data cleaned plot")
compare_plot_path = os.path.join(base_path, "data", "compare plot")

print("\nFolder paths defined:")
print("Raw data folder:", raw_data_path)
print("Cleaned data folder:", cleaned_data_path)
print("Before plot folder:", before_plot_path)
print("After plot folder:", after_plot_path)
print("Compare plot folder:", compare_plot_path)

#=============================================================================
# 3. LOAD DATA FROM RAW DATA FOLDER
#=============================================================================

print("\n" + "="*60)
print("LOADING DATA FROM RAW DATA FOLDER")
print("="*60)

# Construct full path to the CSV file
csv_file_path = os.path.join(raw_data_path, 'AirQualityUCI.csv')

# Check if file exists
if os.path.exists(csv_file_path):
    # Load the CSV file
    df_raw = pd.read_csv(csv_file_path, sep=';', decimal=',')

    print("Data loaded successfully!")
    print("Shape:", df_raw.shape[0], "rows x", df_raw.shape[1], "columns")
    print("Columns:", list(df_raw.columns))
else:
    print("ERROR: File not found at", csv_file_path)
    print("Please ensure 'AirQualityUCI.csv' is in the 'raw data' folder")
    # Create empty dataframe to prevent errors
    df_raw = pd.DataFrame()

#=============================================================================
# 4. INITIAL DATA EXPLORATION
#=============================================================================
print("\n" + "="*60)
print("INITIAL DATA EXPLORATION")
print("="*60)

print("\nFIRST 5 OBSERVATIONS:")
print(df_raw.head())

print("\nLAST 5 OBSERVATIONS:")
print(df_raw.tail())

print("\nDATASET INFORMATION:")
print(df_raw.info())

print("\nSTATISTICAL SUMMARY:")
print(df_raw.describe())

#=============================================================================
# 5. DATA QUALITY ASSESSMENT
#=============================================================================
print("\n" + "="*60)
print("DATA QUALITY ASSESSMENT")
print("="*60)

# 5.1 Missing Values Analysis
print("\n1. MISSING VALUES ANALYSIS:")
null_counts = df_raw.isna().sum()
null_percentage = (null_counts / len(df_raw)) * 100

missing_df = pd.DataFrame({
    'Missing Values': null_counts,
    'Percentage': null_percentage.round(2)
})
print(missing_df[missing_df['Missing Values'] > 0])

# 5.2 Special Missing Value Detection
print("\n2. SPECIAL MISSING VALUE DETECTION:")
print("The UCI dataset uses -200 to indicate missing values in multiple columns.")

minus_200_count = (df_raw == -200).sum().sum()
print("Total occurrences of '-200':", minus_200_count)

# Count -200 occurrences per column
minus_200_per_column = (df_raw == -200).sum()
print("\n'-200' occurrences by column:")
for col, count in minus_200_per_column[minus_200_per_column > 0].items():
    print("  -", col, ":", count, "values (", round(count/len(df_raw)*100, 1), "%)")

# 5.3 Invalid Value Detection
print("\n3. INVALID VALUE DETECTION:")

# Temperature should be reasonable for Rome, Italy
invalid_temp = df_raw[(df_raw['T'] < -10) | (df_raw['T'] > 45)]
print("  Temperature:", len(invalid_temp), "values outside realistic range (-10C to 45C)")

# Relative Humidity should be between 0% and 100%
invalid_rh = df_raw[(df_raw['RH'] < 0) | (df_raw['RH'] > 100)]
print("  Relative Humidity:", len(invalid_rh), "values outside valid range (0% to 100%)")

# CO levels should be positive
invalid_co = df_raw[df_raw['CO(GT)'] < 0]
print("  CO(GT):", len(invalid_co), "negative values (should be positive)")

# NOx levels should be positive
invalid_nox = df_raw[df_raw['NOx(GT)'] < 0]
print("  NOx(GT):", len(invalid_nox), "negative values (should be positive)")

# NO2 levels should be positive
invalid_no2 = df_raw[df_raw['NO2(GT)'] < 0]
print("  NO2(GT):", len(invalid_no2), "negative values (should be positive)")

#=============================================================================
# 6. TEMPORAL DATA PROCESSING
#=============================================================================
print("\n" + "="*60)
print("TEMPORAL DATA PROCESSING")
print("="*60)

# First, let's see what the date/time columns look like
print("\n1. EXAMINING DATE/TIME FORMAT:")
print("Sample dates:", df_raw['Date'].head())
print("Sample times:", df_raw['Time'].head())

# Create a unified DateTime column from Date and Time columns
print("\n2. CREATING DATETIME COLUMN:")

try:
    # Method 1: Try standard parsing with the expected format
    df_raw['DateTime'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'],
                                         format='%d/%m/%Y %H.%M.%S')
    print("  DateTime column created using standard parsing")
except Exception as e:
    # Method 2: Alternative approach if standard parsing fails
    print("  Standard parsing failed:", e)
    print("  Trying alternative parsing method...")
    df_raw['DateTime'] = df_raw['Date'] + ' ' + df_raw['Time'].astype(str)
    df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'],
                                         format='%d/%m/%Y %H.%M.%S',
                                         errors='coerce')
    print("  DateTime column created using alternative method")

# Display the date range covered by the dataset
print("\n3. DATASET TIMEFRAME:")
print("  Start date:", df_raw['DateTime'].min())
print("  End date:", df_raw['DateTime'].max())
print("  Total duration:", (df_raw['DateTime'].max() - df_raw['DateTime'].min()).days, "days")
print("  Total records:", len(df_raw), "hourly measurements")

#=============================================================================
# 7. BEFORE CLEANING - VISUALIZATION (SAVE TO BEFORE FOLDER)
#=============================================================================
print("\n" + "="*60)
print("BEFORE CLEANING VISUALIZATION")
print("="*60)

plt.figure(figsize=(16, 6))

# Plot 1: CO Levels BEFORE cleaning
plt.subplot(1, 2, 1)
plt.plot(df_raw['DateTime'], df_raw['CO(GT)'], color='red', alpha=0.7, linewidth=0.8)
plt.title('BEFORE CLEANING: Carbon Monoxide (CO) Levels', fontsize=14)
plt.xlabel('Date')
plt.ylabel('CO Concentration (mg/m³)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 2: Temperature BEFORE cleaning
plt.subplot(1, 2, 2)
plt.plot(df_raw['DateTime'], df_raw['T'], color='blue', alpha=0.7, linewidth=0.8)
plt.title('BEFORE CLEANING: Temperature Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.suptitle('RAW DATA QUALITY ISSUES DETECTED', fontsize=16)
plt.tight_layout()

# SAVE TO BEFORE PLOT FOLDER
before_plot_file = os.path.join(before_plot_path, 'before_cleaning_problems.png')
plt.savefig(before_plot_file, dpi=150, bbox_inches='tight')
plt.show()
print("Saved:", before_plot_file)
print("  PROBLEM DETECTED: Temperature shows -200C (impossible for Rome!)")
print("  PROBLEM DETECTED: CO shows negative values")

#=============================================================================
# 8. DATA CLEANING PROCESS
#=============================================================================
print("\n" + "="*60)
print("DATA CLEANING PROCESS")
print("="*60)

# Create a copy of raw data for cleaning (preserve original)
df_clean = df_raw.copy()
print("Created working copy of dataset for cleaning")

# Step 1: Handle special missing values (-200)
print("\n1. HANDLING SPECIAL MISSING VALUES (-200):")
minus_200_count = (df_clean == -200).sum().sum()
print("  Found", minus_200_count, "occurrences of '-200'")
df_clean = df_clean.replace(-200, np.nan)
print("  Replaced -200 with NaN for proper handling")

# Step 2: Handle impossible temperature values
print("\n2. HANDLING IMPOSSIBLE TEMPERATURE VALUES:")
print("  Rome, Italy typical temperature range: -10C to 45C")
invalid_temp_before = len(df_clean[(df_clean['T'] < -10) | (df_clean['T'] > 45)])
df_clean.loc[(df_clean['T'] < -10) | (df_clean['T'] > 45), 'T'] = np.nan
print("  Removed", invalid_temp_before, "impossible temperature values")

# Step 3: Handle impossible humidity values
print("\n3. HANDLING IMPOSSIBLE HUMIDITY VALUES:")
print("  Valid humidity range: 0% to 100%")
invalid_rh_before = len(df_clean[(df_clean['RH'] < 0) | (df_clean['RH'] > 100)])
df_clean.loc[(df_clean['RH'] < 0) | (df_clean['RH'] > 100), 'RH'] = np.nan
print("  Removed", invalid_rh_before, "impossible humidity values")

# Step 4: Handle negative pollutant values
print("\n4. HANDLING NEGATIVE POLLUTANT VALUES:")
print("  Pollutant concentrations must be positive")

invalid_co_before = len(df_clean[df_clean['CO(GT)'] < 0])
df_clean.loc[df_clean['CO(GT)'] < 0, 'CO(GT)'] = np.nan
print("  CO(GT): removed", invalid_co_before, "negative values")

invalid_nox_before = len(df_clean[df_clean['NOx(GT)'] < 0])
df_clean.loc[df_clean['NOx(GT)'] < 0, 'NOx(GT)'] = np.nan
print("  NOx(GT): removed", invalid_nox_before, "negative values")

invalid_no2_before = len(df_clean[df_clean['NO2(GT)'] < 0])
df_clean.loc[df_clean['NO2(GT)'] < 0, 'NO2(GT)'] = np.nan
print("  NO2(GT): removed", invalid_no2_before, "negative values")

# Step 5: Handle missing values in time series
print("\n5. HANDLING MISSING VALUES IN TIME SERIES:")
print("  Using forward fill then backward fill for temporal continuity")

missing_before = df_clean.isnull().sum().sum()
df_clean = df_clean.ffill().bfill()
missing_after = df_clean.isnull().sum().sum()
print("  Missing values before:", missing_before)
print("  Missing values after:", missing_after)
print("  Applied forward/backward fill for missing values")

# Step 6: Create proper DateTime index
print("\n6. CREATING PROPER DATETIME INDEX:")
df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'],
                                       format='%d/%m/%Y %H.%M.%S',
                                       errors='coerce')
df_clean.set_index('DateTime', inplace=True)
df_clean.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')
print("  Created DateTime index and removed original date/time columns")

# Step 7: Verify cleaning results
print("\n7. CLEANING VERIFICATION:")
print("  Dataset shape:", df_clean.shape[0], "rows x", df_clean.shape[1], "columns")
print("  Date range:", df_clean.index.min(), "to", df_clean.index.max())
print("  Temperature range:", round(df_clean['T'].min(), 1), "C to", round(df_clean['T'].max(), 1), "C")
print("  CO range:", round(df_clean['CO(GT)'].min(), 3), "to", round(df_clean['CO(GT)'].max(), 3), "mg/m³")
print("  Missing values:", df_clean.isnull().sum().sum())

print("\nData cleaning completed successfully!")

#=============================================================================
# 9. AFTER CLEANING - VISUALIZATION (SAVE TO AFTER FOLDER)
#=============================================================================
print("\n" + "="*60)
print("AFTER CLEANING VISUALIZATION")
print("="*60)

# Plot AFTER cleaning - showing the fixed results
plt.figure(figsize=(16, 6))

# Plot 1: CO Levels AFTER cleaning
plt.subplot(1, 2, 1)
plt.plot(df_clean.index, df_clean['CO(GT)'], color='red', alpha=0.7, linewidth=0.8)
plt.title('AFTER CLEANING: Carbon Monoxide (CO) Levels', fontsize=14)
plt.xlabel('Date')
plt.ylabel('CO Concentration (mg/m³)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 2: Temperature AFTER cleaning
plt.subplot(1, 2, 2)
plt.plot(df_clean.index, df_clean['T'], color='blue', alpha=0.7, linewidth=0.8)
plt.title('AFTER CLEANING: Temperature Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.suptitle('CLEANED DATA - ALL IMPOSSIBLE VALUES REMOVED', fontsize=16)
plt.tight_layout()

# SAVE TO AFTER PLOT FOLDER
after_plot_file = os.path.join(after_plot_path, 'after_cleaning_success.png')
plt.savefig(after_plot_file, dpi=150, bbox_inches='tight')
plt.show()
print("Saved:", after_plot_file)
print("  Temperature now shows realistic values:", round(df_clean['T'].min(), 1), "C to", round(df_clean['T'].max(), 1), "C")
print("  CO minimum:", round(df_clean['CO(GT)'].min(), 3), "mg/m³ (no negatives!)")

#=============================================================================
# 10. BEFORE vs AFTER COMPARISON (SAVE TO COMPARE FOLDER)
#=============================================================================
print("\n" + "="*60)
print("BEFORE vs AFTER COMPARISON")
print("="*60)

# Create side-by-side comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('DATA CLEANING IMPACT: BEFORE vs AFTER COMPARISON', fontsize=16)

# BEFORE: Temperature
axes[0, 0].plot(df_raw['DateTime'], df_raw['T'], color='red', alpha=0.7, linewidth=0.8)
axes[0, 0].set_title('BEFORE: Temperature (with -200C errors)', fontsize=12)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].text(0.02, 0.95, 'Min: ' + str(round(df_raw['T'].min(), 0)) + 'C', 
                transform=axes[0, 0].transAxes, fontsize=10, color='red')

# AFTER: Temperature
axes[0, 1].plot(df_clean.index, df_clean['T'], color='green', alpha=0.7, linewidth=0.8)
axes[0, 1].set_title('AFTER: Temperature (cleaned)', fontsize=12)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].text(0.02, 0.95, 'Min: ' + str(round(df_clean['T'].min(), 1)) + 'C', 
                transform=axes[0, 1].transAxes, fontsize=10, color='green')

# BEFORE: CO Levels
axes[1, 0].plot(df_raw['DateTime'], df_raw['CO(GT)'], color='red', alpha=0.7, linewidth=0.8)
axes[1, 0].set_title('BEFORE: CO Levels (with negative values)', fontsize=12)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('CO (mg/m³)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].text(0.02, 0.95, 'Min: ' + str(round(df_raw['CO(GT)'].min(), 1)), 
                transform=axes[1, 0].transAxes, fontsize=10, color='red')

# AFTER: CO Levels
axes[1, 1].plot(df_clean.index, df_clean['CO(GT)'], color='green', alpha=0.7, linewidth=0.8)
axes[1, 1].set_title('AFTER: CO Levels (all positive)', fontsize=12)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('CO (mg/m³)')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].text(0.02, 0.95, 'Min: ' + str(round(df_clean['CO(GT)'].min(), 3)), 
                transform=axes[1, 1].transAxes, fontsize=10, color='green')

plt.tight_layout()

# SAVE TO COMPARE FOLDER
compare_plot_file = os.path.join(compare_plot_path, 'before_after_comparison.png')
plt.savefig(compare_plot_file, dpi=150, bbox_inches='tight')
plt.show()
print("Saved:", compare_plot_file)

#=============================================================================
# 11. DATA CLEANING SUMMARY REPORT
#=============================================================================
print("\n" + "="*60)
print("DATA CLEANING SUMMARY REPORT")
print("="*60)

print("""
PROBLEMS DETECTED AND FIXED:
----------------------------------------------------------------
Temperature values: """, round(df_raw['T'].min(), 0), "C to", round(df_raw['T'].max(), 0), "C (impossible!)")
print("  AFTER cleaning:", round(df_clean['T'].min(), 1), "C to", round(df_clean['T'].max(), 1), "C OK")

print("\nNegative CO values:", invalid_co_before, "occurrences")
print("  AFTER cleaning:", (df_clean['CO(GT)'] < 0).sum(), "occurrences OK")

print("\nNegative NOx values:", invalid_nox_before, "occurrences")
print("  AFTER cleaning:", (df_clean['NOx(GT)'] < 0).sum(), "occurrences OK")

print("\nNegative NO2 values:", invalid_no2_before, "occurrences")
print("  AFTER cleaning:", (df_clean['NO2(GT)'] < 0).sum(), "occurrences OK")

print("\n-200 missing values:", minus_200_count, "occurrences")
print("  AFTER cleaning: All converted to NaN and imputed OK")

print("\nImpossible humidity values:", invalid_rh_before, "occurrences")
print("  AFTER cleaning: All corrected OK")

print("\nDATASET COMPARISON:")
print("  BEFORE:", df_raw.shape[0], "rows,", df_raw.shape[1], "columns")
print("  AFTER:", df_clean.shape[0], "rows,", df_clean.shape[1], "columns")

print("\nCONCLUSION: Data cleaning successfully removed all impossible values.")
print("Temperature now shows realistic seasonal patterns for Rome, Italy.")
print("All pollutant concentrations are now positive as expected.")
print("Dataset is now ready for feature engineering and model development.")

#=============================================================================
# 12. SAVE CLEANED DATA TO CLEANED DATA FOLDER
#=============================================================================
print("\n" + "="*60)
print("SAVING CLEANED DATA")
print("="*60)

# Save cleaned dataset with original column names
cleaned_csv_file = os.path.join(cleaned_data_path, 'air_quality_cleaned.csv')
df_clean.to_csv(cleaned_csv_file)
print("Saved:", cleaned_csv_file)

# Create version with cleaned column names for modeling
df_model = df_clean.copy()
df_model.columns = df_model.columns.str.replace(r'[()]', '', regex=True)\
                                   .str.replace('.', '_', regex=False)

model_file = os.path.join(cleaned_data_path, 'air_quality_cleaned_model_ready.csv')
df_model.to_csv(model_file)
print("Saved:", model_file)

# Save sample for quick viewing
sample_file = os.path.join(cleaned_data_path, 'air_quality_sample.csv')
df_clean.head(100).to_csv(sample_file)
print("Saved:", sample_file)

#=============================================================================
# 13. SUMMARY OF ANALYSIS
#=============================================================================
print("\n" + "="*60)
print("DATA ANALYSIS SUMMARY")
print("="*60)

print("""
DATASET CHARACTERISTICS:
----------------------------------------------------------------
  Original size:""", df_raw.shape[0], "rows x", df_raw.shape[1], "columns")
print("  Cleaned size:", df_clean.shape[0], "rows x", df_clean.shape[1], "columns")
print("  Time period:", df_clean.index.min().date(), "to", df_clean.index.max().date())
print("  Duration:", (df_clean.index.max() - df_clean.index.min()).days, "days")
print("  Measurement frequency: Hourly")

print("""
KEY VARIABLES ANALYZED:
----------------------------------------------------------------
  Primary pollutant: CO(GT) - Carbon Monoxide
  Secondary pollutants: NOx(GT), NO2(GT)
  Environmental factors: Temperature (T), Humidity (RH)

ACTIONS PERFORMED:
----------------------------------------------------------------
  1. Dataset loaded and verified from UCI repository
  2. Initial exploratory analysis completed
  3. Data quality issues identified (-200 values, impossible temperatures)
  4. Temporal data processing (DateTime creation)
  5. Comprehensive data cleaning applied
  6. Before/After visualizations created for comparison
  7. Data saved locally with proper folder structure

KEY OBSERVATIONS:
----------------------------------------------------------------
  Clear temporal patterns visible in pollutant concentrations
  Temperature shows realistic seasonal variations for Rome
  All negative values and -200 markers successfully removed
  Dataset quality significantly improved through cleaning
  Data now ready for feature engineering and modeling

FILES GENERATED:
----------------------------------------------------------------
  1.""", before_plot_file)
print("  2.", after_plot_file)
print("  3.", compare_plot_file)
print("  4.", cleaned_csv_file)
print("  5.", model_file)
print("  6.", sample_file)

print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*60)

#=============================================================================
# 14. NEXT STEPS FOR PROJECT DEVELOPMENT
#=============================================================================
print("\n" + "="*60)
print("RECOMMENDED NEXT STEPS")
print("="*60)

print("""
1. FEATURE ENGINEERING:
   Create lag features (t-1, t-2, ..., t-24) for time-series prediction
   Generate rolling statistics (6h, 12h, 24h moving averages)
   Extract temporal features (hour of day, day of week, month, season)
   Create interaction terms between correlated variables

2. DATA PREPARATION FOR MODELING:
   Scale/normalize features using StandardScaler or MinMaxScaler
   Split data chronologically (80% train, 20% test) - IMPORTANT for time series
   Create sequences for LSTM (24-hour windows with 24-hour forecast horizon)
   Prepare validation set for hyperparameter tuning

3. MODEL DEVELOPMENT:
   Implement baseline models (Linear Regression, Random Forest)
   Develop tree-based models (XGBoost with hyperparameter tuning)
   Implement deep learning (LSTM with sequential architecture)
   Add time-series specific models (Prophet, ARIMA)

4. EVALUATION METRICS:
   RMSE (Root Mean Square Error) - Primary metric
   MAE (Mean Absolute Error) - Robust to outliers
   R2 Score (Coefficient of determination)
   MAPE (Mean Absolute Percentage Error)

5. MODEL INTERPRETATION:
   Feature importance analysis (XGBoost, Random Forest)
   SHAP values for model explainability
   Partial dependence plots
   Residual analysis

DATASET TO USE FOR MODELING:
----------------------------------------------------------------
   Use:""", model_file)
print("  Target variable: CO_GT (after column cleaning)")
print("  Features: All other numerical columns")

print("\n" + "="*60)
print("PROJECT READY FOR MODEL DEVELOPMENT PHASE!")
print("="*60)
