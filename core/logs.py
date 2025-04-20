import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

def log_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Log the metrics
    mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae})