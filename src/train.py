from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mse': mean_squared_error(y_test, y_pred_test)
    }
    
    return model, y_pred_test, metrics