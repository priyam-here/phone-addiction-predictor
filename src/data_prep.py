import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Name', 'Location'])
    
    categorical_cols = ['Gender', 'Phone_Usage_Purpose']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('Addiction_Level', axis=1)
    y = df_encoded['Addiction_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, df_encoded