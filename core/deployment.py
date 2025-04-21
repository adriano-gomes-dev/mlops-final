import pandas as pd
import mlflow

from mlflow.tracking import MlflowClient

def promote_model_to_production_based_on_mse():
    client = MlflowClient()
    list_metrics_version = []
    # List all versions of a model
    for mv in client.search_model_versions("name='incc_model'"):
        print(f"Version: {mv.version}, Stage: {mv.current_stage}")

        # Step 1: Get run_id from model version
        model_version_info = client.get_model_version(name=mv.name, version=mv.version)
        run_id = model_version_info.run_id

        # Step 2: Get metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics['mse']

        list_metrics_version.append({'version': mv.version, 'mse': metrics})
        
    df_metrics_version = pd.DataFrame(list_metrics_version)
    min_mse = df_metrics_version['mse'].min()
    min_mse_version = df_metrics_version[df_metrics_version['mse'] == min_mse]['version'].values[0]
    print(f"Versão com o menor MSE: {min_mse_version}")

    client.transition_model_version_stage(
        name=mv.name,
        version=min_mse_version,
        stage="Production"
    )

def get_prediction_from_production_model(value_to_predict):
    # Set the tracking URI to the local MLflow server
    # precisa chamar aqui novamente por causa do fastapi
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    model_production = None

    for mv in client.search_model_versions("name='incc_model'"):
        print(f"Version: {mv.version}, Stage: {mv.current_stage}")

        # # Step 1: Get run_id from model version
        # model_version_info = client.get_model_version(name=mv.name, version=mv.version)
        # run_id = model_version_info.run_id
        
        if mv.current_stage == "Production":
            model_production = mlflow.pyfunc.load_model(model_uri=f"models:/{mv.name}/{mv.version}")
            break
    
    data_to_predict = pd.DataFrame({'Data': [value_to_predict]})
    predictions = model_production.predict(data_to_predict)
    return str(predictions[0][0])