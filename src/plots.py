import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def generate_visualizations(y_test, y_pred, model, feature_names, raw_df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Actual vs Predicted (Scatter)
   
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted Phone Addiction Level (Uncapped)')
    plt.xlabel('Actual Addiction Level (Ground Truth)')
    plt.ylabel('Predicted Addiction Level (Model Output)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_actual_vs_predicted.png')
    plt.close()
    
    # Plot 2: Feature Importance (Bar Chart)

    coefs = pd.Series(model.coef_, index=feature_names)
    top_coefs = coefs.abs().sort_values(ascending=False).head(10)
    actual_top_coefs = coefs[top_coefs.index].sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if c < 0 else 'green' for c in actual_top_coefs]
    actual_top_coefs.plot(kind='barh', color=colors)
    plt.title('Top 10 Feature Importances (Linear Regression)')
    plt.xlabel('Coefficient Value (Impact on Addiction Level)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_feature_importance.png')
    plt.close()

    # Plot 3: Residuals Distribution (Error Histogram)

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('Distribution of Model Errors (Residuals)')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_residual_distribution.png')
    plt.close()

    # Plot 4: The Data Discovery (Capped vs Uncapped)

    plt.figure(figsize=(12, 5))
    
    # Left subplot: The raw, capped data
    plt.subplot(1, 2, 1)
    plt.hist(raw_df['Addiction_Level'], bins=20, color='orange', edgecolor='black')
    plt.title('Original Data (Cap at 10 is evident)')
    plt.xlabel('Addiction Level')
    plt.ylabel('Count')
    
    # Right subplot: The cleaned data
    uncapped_data = raw_df[raw_df['Addiction_Level'] < 10]['Addiction_Level']
    plt.subplot(1, 2, 2)
    plt.hist(uncapped_data, bins=20, color='teal', edgecolor='black')
    plt.title('Cleaned Data (Cap Removed)')
    plt.xlabel('Addiction Level')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_data_discovery.png')
    plt.close()

def plot_correlation_heatmap(df, output_dir='outputs'):
    # Generates a post-processing EDA heatmap of feature correlations
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    
    # Calculate correlations
    correlation_matrix = df.corr()
    
    # Isolate correlations with the target, drop the target itself from the rows, and sort
    target_corr = correlation_matrix[['Addiction_Level']].drop('Addiction_Level')
    target_corr = target_corr.sort_values(by='Addiction_Level', ascending=False)
    
    # Draw the heatmap
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    
    plt.title('Feature Correlation with Addiction Level')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/0_correlation_heatmap.png')
    plt.close()
