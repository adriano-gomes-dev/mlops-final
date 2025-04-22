import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from core.logs import log_metrics

def experiment_linear_regression(X_train, X_test, y_train, y_test):

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        log_metrics(y_test, y_pred)

        print(f"Run ID: {run.info.run_id}")

        mlflow.sklearn.log_model(model, "linear_regression_model", registered_model_name="incc_model")
        print(f"Modelo Linear Regression registrado no MLflow! Run ID: {run.info.run_id}")

def experiment_svr(X_train, X_test, y_train, y_test):
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "Linear SVR")

        model = SVR(kernel='linear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        log_metrics(y_test, y_pred)

        print(f"Run ID: {run.info.run_id}")

        mlflow.sklearn.log_model(model, "linear_SVR_model", registered_model_name="incc_model")
        print(f"Modelo Linear SRV registrado no MLflow! Run ID: {run.info.run_id}")

def experiment_random_forest_regressor(X_train, X_test, y_train, y_test):
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "Random Forest Regressor")

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        
        log_metrics(y_test, y_pred)

        print(f"Run ID: {run.info.run_id}")

        mlflow.sklearn.log_model(model, "linear_random_forest_regressor", registered_model_name="incc_model")
        print(f"Modelo Linear Random Forest Regressor registrado no MLflow! Run ID: {run.info.run_id}")
