"""
visualizer.py - Advanced Visualization (Publication Quality)

This module contains all plotting functions for the AQI Forecasting Project.
Each function creates publication-quality visualizations for the final report.

Functions:
1. plot_model_comparison() - Bar charts comparing all models
2. plot_predictions() - Actual vs predicted with error shading
3. plot_error_distribution() - Histogram of prediction errors
4. plot_hyperparameter_comparison() - Before/after tuning comparison
5. plot_residual_analysis() - Residuals over time and histogram
6. plot_feature_importance() - Top N feature importance horizontal bars
7. plot_full_dashboard() - Complete dashboard with all plots combined
8. plot_residual_vs_predicted() - Scatter plot of residuals vs predicted values

Author: Dhanunjaya Rao Thandra
Date: April 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from .config import PLOTS_PATH

logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL PLOT STYLE SETTINGS
# =============================================================================
# These settings ensure all plots have a consistent, professional look
plt.rcParams.update({
    "figure.facecolor": "white",           # White background for figures
    "axes.facecolor": "white",             # White background for axes
    "axes.edgecolor": "#333333",           # Dark gray axis lines
    "axes.labelcolor": "#333333",          # Dark gray axis labels
    "xtick.color": "#333333",              # Dark gray tick marks
    "ytick.color": "#333333",              # Dark gray tick marks
    "font.size": 11,                       # Base font size
    "axes.titleweight": "bold"             # Bold axis titles
})


# =============================================================================
# 1. MODEL COMPARISON PLOT
# =============================================================================
def plot_model_comparison(results_df):
    """
    Create bar charts comparing model performance (RMSE and R²)
    
    This function creates two side-by-side bar charts:
    - Left: RMSE values (lower is better)
    - Right: R² scores (higher is better)
    
    The best model in each chart is highlighted in green.
    
    Args:
        results_df: DataFrame with columns 'Model', 'RMSE', 'R2'
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    models = results_df['Model']
    rmse = results_df['RMSE']
    r2 = results_df['R2']

    # Find indices of best performing models
    best_rmse_idx = rmse.idxmin()   # Lowest RMSE is best
    best_r2_idx = r2.idxmax()       # Highest R² is best

    # Create color arrays - green for best model, gray for others
    colors_rmse = ['#d3d3d3'] * len(models)
    colors_r2 = ['#d3d3d3'] * len(models)
    colors_rmse[best_rmse_idx] = '#27ae60'  # Green for best RMSE
    colors_r2[best_r2_idx] = '#27ae60'      # Green for best R²

    # ---- LEFT PLOT: RMSE Comparison ----
    bars1 = axes[0].bar(models, rmse, color=colors_rmse)
    axes[0].set_title("RMSE Comparison (Lower = Better)", fontsize=14)
    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Add value labels on top of each bar
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height,
                     f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    # ---- RIGHT PLOT: R² Comparison ----
    bars2 = axes[1].bar(models, r2, color=colors_r2)
    axes[1].set_title("R² Score Comparison (Higher = Better)", fontsize=14)
    axes[1].set_ylabel("R² Score", fontsize=12)
    axes[1].set_ylim(0, 1)  # R² is always between 0 and 1
    axes[1].tick_params(axis='x', rotation=45)

    # Add value labels on top of each bar
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height,
                     f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle("Model Performance Overview", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save high-resolution image
    save_path = PLOTS_PATH / 'model_comparison_advanced.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return fig


# =============================================================================
# 2. PREDICTION PLOT (ACTUAL VS PREDICTED)
# =============================================================================
def plot_predictions(y_true, y_pred, model_name):
    """
    Create actual vs predicted plot with error shading
    
    This function shows how well the model's predictions match actual values.
    The shaded area between the lines represents the prediction error.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model (for title)
    
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(16, 6))

    # Plot first 300 points for clarity (full series would be too crowded)
    pts = min(300, len(y_true))
    x = np.arange(pts)

    y_true_plot = np.array(y_true[:pts])
    y_pred_plot = np.array(y_pred[:pts])

    # Main lines
    plt.plot(x, y_true_plot, label="Actual", linewidth=2.5, color='#2E86AB')
    plt.plot(x, y_pred_plot, label="Predicted", linestyle='--', linewidth=2, color='#2ECC71')

    # Error shading - shows the gap between actual and predicted
    plt.fill_between(x, y_true_plot, y_pred_plot, alpha=0.2, color='gray')

    plt.title(f"{model_name}: Actual vs Predicted CO Levels", fontsize=15, fontweight='bold')
    plt.xlabel("Time Index (Hours)", fontsize=12)
    plt.ylabel("CO Concentration (mg/m³)", fontsize=12)

    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_path = PLOTS_PATH / 'predictions_advanced.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return plt.gcf()


# =============================================================================
# 3. ERROR DISTRIBUTION PLOT
# =============================================================================
def plot_error_distribution(y_true, y_pred):
    """
    Create histogram of prediction errors
    
    This plot shows the distribution of errors (Actual - Predicted).
    Ideally, errors should be centered around zero (no systematic bias)
    and follow a normal distribution.
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        matplotlib figure object
    """
    errors = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(10, 6))

    # Histogram with 50 bins for detailed distribution view
    plt.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')

    # Add mean and standard deviation to the plot
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='green', linestyle='--', linewidth=2,
                label=f'Mean: {mean_error:.4f}')

    plt.title("Prediction Error Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Error (Actual - Predicted)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = PLOTS_PATH / 'error_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return plt.gcf()


# =============================================================================
# 4. HYPERPARAMETER TUNING COMPARISON PLOT
# =============================================================================
def plot_hyperparameter_comparison(default_metrics, tuned_metrics):
    """
    Create before/after comparison of hyperparameter tuning
    
    This plot shows how tuning improved model performance.
    
    Args:
        default_metrics: Dict with 'RMSE' and 'R2' for default model
        tuned_metrics: Dict with 'RMSE' and 'R2' for tuned model
    
    Returns:
        matplotlib figure object
    """
    categories = ['RMSE', 'R²']
    before = [default_metrics['RMSE'], default_metrics['R2']]
    after = [tuned_metrics['RMSE'], tuned_metrics['R2']]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, before, width, label="Default XGBoost", alpha=0.8, color='#95A5A6')
    bars2 = ax.bar(x + width/2, after, width, label="Tuned XGBoost", alpha=0.8, color='#2ECC71')

    ax.set_title("Hyperparameter Tuning Impact", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement percentage labels
    for i in range(len(categories)):
        improvement = ((after[i] - before[i]) / before[i]) * 100
        sign = '+' if improvement > 0 else ''
        color = '#27ae60' if improvement > 0 else '#e74c3c'
        ax.text(i, max(before[i], after[i]) + 0.01,
                f"{sign}{improvement:.1f}%", ha='center', fontsize=11,
                fontweight='bold', color=color)

    # Add value labels on bars
    for bar, val in zip(bars1, before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = PLOTS_PATH / 'tuning_advanced.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return fig


# =============================================================================
# 5. RESIDUAL ANALYSIS PLOT
# =============================================================================
def plot_residual_analysis(y_true, y_pred, model_name):
    """
    Create residual analysis plot to check for systematic bias
    
    This function creates two subplots:
    - Left: Residuals over time (should be randomly scattered around zero)
    - Right: Histogram of residuals (should be approximately normal)
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model
    
    Returns:
        matplotlib figure object
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} - Residual Analysis", fontsize=14, fontweight='bold')

    # ----- LEFT PLOT: Residuals over time -----
    axes[0].plot(residuals, alpha=0.6, color='#9B59B6', linewidth=1)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title("Residuals Over Time", fontsize=12)
    axes[0].set_xlabel("Time Index", fontsize=10)
    axes[0].set_ylabel("Residual", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ----- RIGHT PLOT: Residual histogram -----
    axes[1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].axvline(np.mean(residuals), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(residuals):.4f}')
    axes[1].set_title("Distribution of Residuals", fontsize=12)
    axes[1].set_xlabel("Residual", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = PLOTS_PATH / 'residual_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return fig


# =============================================================================
# 6. FEATURE IMPORTANCE PLOT
# =============================================================================
def plot_feature_importance(model, feature_names, top_n=15, model_name="XGBoost"):
    """
    Plot top N feature importance as horizontal bar chart
    
    This function shows which features were most important for predictions.
    Longer bars indicate more important features.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        model_name: Name of the model
    
    Returns:
        importance_df: DataFrame with features and their importance scores
    """
    importances = model.feature_importances_

    # Create dataframe and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))

    # Create horizontal bar chart
    bars = plt.barh(importance_df['feature'], importance_df['importance'],
                    color='teal', edgecolor='black', alpha=0.8)

    # Highlight the top feature in green
    bars[0].set_color('#27ae60')

    plt.xlabel("Importance", fontsize=12, fontweight='bold')
    plt.title(f"{model_name} - Top {top_n} Feature Importance", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    save_path = PLOTS_PATH / 'feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return importance_df


# =============================================================================
# 7. RESIDUAL VS PREDICTED SCATTER PLOT
# =============================================================================
def plot_residual_vs_predicted(y_true, y_pred, model_name):
    """
    Create residual vs predicted scatter plot to check for heteroscedasticity
    
    This plot helps detect patterns in errors:
    - X-axis: Predicted values
    - Y-axis: Residuals (Actual - Predicted)
    - Ideally: Points randomly scattered around zero (no pattern)
    - Problem signs: Funnel shape (heteroscedasticity) or curved pattern
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model
    
    Returns:
        matplotlib figure object
    """
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(10, 7))

    # Scatter plot of residuals vs predicted values
    plt.scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue', edgecolor='black')

    # Zero line (perfect prediction)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero error line')

    # Add trend line to check for patterns
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(y_pred)
    plt.plot(x_sorted, p(x_sorted), color='orange', linewidth=2,
             label=f'Trend line (slope: {z[0]:.4f})')

    # Labels and title
    plt.xlabel("Predicted CO Concentration (mg/m³)", fontsize=12, fontweight='bold')
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12, fontweight='bold')
    plt.title(f"{model_name} - Residuals vs Predicted Values", fontsize=14, fontweight='bold')

    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)

    # Add statistics box
    stats_text = (f"Mean residual: {np.mean(residuals):.4f}\n"
                  f"Std deviation: {np.std(residuals):.4f}\n"
                  f"Variance: {np.var(residuals):.4f}")
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = PLOTS_PATH / 'residual_vs_predicted.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return plt.gcf()


# =============================================================================
# 8. COMPLETE ANALYSIS DASHBOARD
# =============================================================================
def plot_full_dashboard(results_df, y_test, y_pred_best, best_model_name,
                        default_metrics, tuned_metrics, feature_importance_df=None):
    """
    Create a complete analysis dashboard with all plots in one figure
    
    This function combines 6 key plots into a single figure for a comprehensive
    overview of the project results.
    
    Args:
        results_df: DataFrame with model comparison results
        y_test: Actual test values
        y_pred_best: Predictions from best model
        best_model_name: Name of the best model
        default_metrics: Metrics for default XGBoost
        tuned_metrics: Metrics for tuned XGBoost
        feature_importance_df: DataFrame with feature importance (optional)
    
    Returns:
        matplotlib figure object
    """
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("AQI Forecasting - Complete Analysis Dashboard", fontsize=18, fontweight='bold')

    # ----- 1. Model Comparison (RMSE) -----
    ax1 = fig.add_subplot(3, 2, 1)
    models = results_df['Model']
    rmse = results_df['RMSE']
    colors = ['#27ae60' if i == 0 else '#d3d3d3' for i in range(len(models))]
    bars = ax1.bar(models, rmse, color=colors, edgecolor='black')
    ax1.set_title("RMSE Comparison (Lower is Better)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RMSE", fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    for bar, val in zip(bars, rmse):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # ----- 2. Model Comparison (R²) -----
    ax2 = fig.add_subplot(3, 2, 2)
    r2 = results_df['R2']
    bars = ax2.bar(models, r2, color=colors, edgecolor='black')
    ax2.set_title("R² Score Comparison (Higher is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("R² Score", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    for bar, val in zip(bars, r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # ----- 3. Predictions Plot -----
    ax3 = fig.add_subplot(3, 2, 3)
    pts = min(200, len(y_test))
    x = np.arange(pts)
    ax3.plot(x, y_test[:pts], label='Actual', linewidth=2, color='#2E86AB')
    ax3.plot(x, y_pred_best[:pts], label='Predicted', linestyle='--', linewidth=2, color='#2ECC71')
    ax3.fill_between(x, y_test[:pts], y_pred_best[:pts], alpha=0.2, color='gray')
    ax3.set_title(f"{best_model_name}: Actual vs Predicted", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time Index", fontsize=10)
    ax3.set_ylabel("CO Concentration", fontsize=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ----- 4. Error Distribution -----
    ax4 = fig.add_subplot(3, 2, 4)
    errors = y_test - y_pred_best
    ax4.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_title("Error Distribution", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Prediction Error", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # ----- 5. Tuning Comparison -----
    ax5 = fig.add_subplot(3, 2, 5)
    categories = ['RMSE', 'R²']
    before = [default_metrics['RMSE'], default_metrics['R2']]
    after = [tuned_metrics['RMSE'], tuned_metrics['R2']]
    x = np.arange(len(categories))
    width = 0.35
    ax5.bar(x - width/2, before, width, label='Default', alpha=0.8, color='#95A5A6')
    ax5.bar(x + width/2, after, width, label='Tuned', alpha=0.8, color='#2ECC71')
    ax5.set_title("Hyperparameter Tuning Impact", fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    for i, (b, a) in enumerate(zip(before, after)):
        improvement = ((a - b) / b) * 100
        sign = '+' if improvement > 0 else ''
        ax5.text(i, max(b, a) + 0.01, f'{sign}{improvement:.1f}%', ha='center', fontsize=9)

    # ----- 6. Feature Importance -----
    ax6 = fig.add_subplot(3, 2, 6)
    if feature_importance_df is not None:
        top_features = feature_importance_df.head(10)
        ax6.barh(top_features['feature'], top_features['importance'], color='teal', edgecolor='black')
        ax6.set_title("Top 10 Feature Importance", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Importance", fontsize=10)
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
    else:
        ax6.text(0.5, 0.5, "Feature importance not available", ha='center', va='center')
        ax6.set_title("Feature Importance", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = PLOTS_PATH / 'analysis_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Saved: {save_path}")
    return fig
