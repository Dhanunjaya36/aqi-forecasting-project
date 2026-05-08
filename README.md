# AQI Forecasting: 24-Hour Air Quality Prediction Using Machine Learning

**MSc Data Science Individual Project**  
**University of Hertfordshire** – Module 7PAM2002  
**Supervisor:** Klaas Wiersema  
**Author:** Dhanunjaya Rao Thandra (SRN: 23096219)  
**Date:** April 2026

---

## 1. Project Overview

Air pollution is responsible for approximately seven million premature deaths each year (WHO, 2021). This project investigates whether machine learning can accurately forecast the Air Quality Index (AQI) – specifically carbon monoxide (CO) levels – **24 hours in advance** using historical environmental data from Rome, Italy.

Six regression models were implemented and compared:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- **XGBoost** (best performer)

After extensive data cleaning and feature engineering (44 features including lags up to 24 hours, rolling statistics, and cyclical encoding), the **tuned XGBoost** model achieved:
- **RMSE = 0.3109**
- **R² = 0.9436** (explains 94.4% of variance)
- **MAPE = 14.89%**

This represents a **31% improvement** over the baseline neural network reported by De Vito et al. (2008) on the same dataset.

---

## 2. Research Question

> *"Which machine learning model provides the most accurate 24-hour ahead AQI forecasts using historical environmental data from Rome, Italy?"*

---

## 3. Dataset

| Attribute | Description |
|-----------|-------------|
| **Source** | UCI Machine Learning Repository – [Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+quality) |
| **Location** | Rome, Italy (urban traffic-oriented sensor) |
| **Time period** | March 2004 – February 2005 (one full year) |
| **Frequency** | Hourly measurements |
| **Original size** | 9,471 rows × 15 columns |
| **After cleaning** | 9,447 rows × 15 columns |
| **Target variable** | `CO(GT)` – Carbon monoxide concentration (mg/m³) |

**Key variables used:** CO, NOx, NO₂, temperature (T), relative humidity (RH), and several sensor responses.

---

## 4. Project Structure
AQI-Forecasting-Project/
│
├── data/
│   ├── raw/air_quality.csv
│   └── processed/
│
├── src/  
│   ├── data_exploration.py
│   ├── preprocessor.py
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── prophet_model.py
│   └── evaluator.py
│
├── notebooks/  
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── results/
│   ├── plots/
│   └── metrics.csv
│
├── tests/  (Your unit tests)
│
├── requirements.txt
└── README.md  
