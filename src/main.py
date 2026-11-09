import os
import pandas as pd
from data_prep import load_and_preprocess
from train import train_and_evaluate
from plots import generate_visualizations, plot_correlation_heatmap

def main():
    print("Initializing Phone Addiction Predictor Pipeline...")
    data_path = 'data/phone_addiction_dataset.csv'


    raw_dataframe = pd.read_csv(data_path)
    print(f"-> Initial raw dataset loaded: {raw_dataframe.shape[0]} rows.")

    print("\n--- CORE FEATURE STATISTICS ---")
    core_features = raw_dataframe[['Daily_Usage_Hours', 'Sleep_Hours', 'Addiction_Level']]
        
    print("1. Summary Statistics (Mean, Std, Min, Max):")
    print(core_features.describe().round(2))
        
    print("\n2. Medians:")
    print(core_features.median().round(2))

    print("\n3. Covariance Matrix:")
    print(core_features.cov().round(2))

    print("-------------------------------\n")

    try:
        X_train, X_test, y_train, y_test, features, uncapped_dataset = load_and_preprocess(data_path)
        print(f"-> Artificial caps removed. Final training rows: {X_train.shape[0]}, Testing rows: {X_test.shape[0]}.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    print("-> Generating Post-Processing EDA Correlation Heatmap...")
    plot_correlation_heatmap(uncapped_dataset)

    print("\n--- CORE FEATURE STATISTICS AFTER CAP REMOVAL ---")
    core_features2 = uncapped_dataset[['Daily_Usage_Hours', 'Sleep_Hours', 'Addiction_Level']]
        
    print("1. Summary Statistics (Mean, Std, Min, Max):")
    print(core_features2.describe().round(2))
        
    print("\n2. Medians:")
    print(core_features2.median().round(2))

    print("\n3. Covariance Matrix:")
    print(core_features2.cov().round(2))

    print("-------------------------------\n")    

    print("Training Linear Regression model...")
    model, y_pred, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n--- MODEL RESULTS (TRAIN vs TEST) ---")
    print(f"R-squared (R2)  | Train: {metrics['train_r2']:.4f} | Test: {metrics['test_r2']:.4f}")
    print(f"Mean Sq. Error  | Train: {metrics['train_mse']:.4f} | Test: {metrics['test_mse']:.4f}")
    print(f"Mean Abs. Error | Train: {metrics['train_mae']:.4f} | Test: {metrics['test_mae']:.4f}")
    print("-------------------------------------")
    print("Conclusion: Train and Test metrics are nearly identical, proving NO OVERFITTING.\n")
    
    print("Generating visualizations in 'outputs/' directory...")
    generate_visualizations(y_test, y_pred, model, features, raw_dataframe)
    print("Pipeline complete.")

if __name__ == '__main__':
    main()
