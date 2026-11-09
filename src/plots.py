import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_visualizations(y_test, y_pred, model, feature_names, raw_df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted Phone Addiction Level')
    plt.savefig(f'{output_dir}/1_actual_vs_predicted.png')
    plt.close()
    
    # Plot 2: Feature Importance
    coefs = pd.Series(model.coef_, index=feature_names)
    top_coefs = coefs.abs().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    coefs[top_coefs.index].sort_values().plot(kind='barh', color='blue')
    plt.title('Top 10 Feature Importances')
    plt.savefig(f'{output_dir}/2_feature_importance.png')
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
