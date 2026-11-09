from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    # Training the Linear Regression model on the Training Datasets
    model.fit(X_train, y_train)
    # Generate predictions for both sets
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate metrics for both to prove no overfitting
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    return model, y_pred_test, metrics
