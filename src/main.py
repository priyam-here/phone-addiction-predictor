import os
import pandas as pd
from data_prep import load_and_preprocess
from train import train_and_evaluate
from plots import generate_visualizations

def main():
    print("Starting Pipeline...")
    data_path = 'data/phone_addiction_dataset.csv'
    
    X_train, X_test, y_train, y_test, features, df_encoded = load_and_preprocess(data_path)
    
    model, y_pred, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print(f"Test R2: {metrics['test_r2']:.4f}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    
    raw_df = pd.read_csv(data_path)
    generate_visualizations(y_test, y_pred, model, features, raw_df)

if __name__ == '__main__':
    main()