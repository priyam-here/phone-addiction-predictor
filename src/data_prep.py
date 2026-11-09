import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Name', 'Location'])
    
    # Remove artificial cap of Addiction Level 10
    df_uncapped = df[df['Addiction_Level'] < 10].copy()
    
    # Convert Categories to Numerical Data for the model to utilize in training
    categorical_cols = ['Gender', 'Phone_Usage_Purpose']
    df_encoded = pd.get_dummies(df_uncapped, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('Addiction_Level', axis=1)
    y = df_encoded['Addiction_Level']
    
    # Splitting the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scaling the dataset so that numerically larger data columns do not dominate other factors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns,df_encoded
