#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análise de dados do INCC (Índice Nacional de Custo da Construção)
"""

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objects as go

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.svm import SVR

# API - FastAPI
# prescisa instalar o fastapi
# pip install fastapi[standard]
# e rodar com o comando: python -m fastapi dev main.py

import fastapi

app = fastapi.FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Predição de INCC. API para predição de INCC com base na data."}

@app.get("/predict/{date}")
async def predict(date: str):
    return {"date to predict": get_prediction_from_production_model(date)}

@app.get("/luigi/{var}")
async def luigi(var: str):
    return {"var": var}

def log_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Log the metrics
    mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae})

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

        # Step 1: Get run_id from model version
        model_version_info = client.get_model_version(name=mv.name, version=mv.version)
        run_id = model_version_info.run_id

        if mv.current_stage == "Production":
            model_production = mlflow.pyfunc.load_model(model_uri=f"models:/{mv.name}/{mv.version}")
            break
    
    data_to_predict = pd.DataFrame({'Data': [value_to_predict]})
    predictions = model_production.predict(data_to_predict)
    return str(predictions[0][0])

def main():
    # Lendo o arquivo de dados
    df = pd.read_csv('./data/dataset INCC.csv', sep="\t")
    # df = pd.read_csv('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv', sep="\t")
    
    # Exibindo as primeiras linhas do DataFrame
    print("Primeiras linhas do DataFrame:")
    print(df.head())
    
    # Informações básicas sobre o DataFrame
    print("\nInformações do DataFrame:")
    print(df.info())
    
    # Convertendo a coluna 'INCC Geral' para float
    df['INCC Geral float'] = df['INCC Geral'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
    
    # Removendo colunas desnecessárias
    df.drop(columns=['INCC Geral'], inplace=True)
    
    print("\nDataFrame após processamento:")
    print(df.head())

    # MLFlow Conexão Info
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("INCC Tracking")


    # Preparando os dados para o modelo
    X = df[['Data']]
    y = df[['INCC Geral float']]

    # Split treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nDimensões dos conjuntos de dados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}") 
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Experimento 1: Regressão Linear
    experiment_linear_regression(X_train, X_test, y_train, y_test)

    # Experimento 2: Linear SVR
    experiment_svr(X_train, X_test, y_train, y_test)

    # Promover o modelo com o menor MSE para a produção
    promote_model_to_production_based_on_mse()

    # todo: refatorar o código para que o modelo seja carregado apenas uma vez


if __name__ == "__main__":
    main() 

