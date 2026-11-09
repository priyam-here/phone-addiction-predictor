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