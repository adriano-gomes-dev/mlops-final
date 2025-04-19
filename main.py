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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def experiment_linear_regression(X_train, X_test, y_train, y_test):

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        print(f"Run ID: {run.info.run_id}")

        mlflow.sklearn.log_model(model, "linear_regression_model")
        print(f"Modelo Linear Regression registrado no MLflow! Run ID: {run.info.run_id}")

def experiment_svr(X_train, X_test, y_train, y_test):
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "Linear SVR")

        model = SVR(kernel='linear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        print(f"Run ID: {run.info.run_id}")

        mlflow.sklearn.log_model(model, "linear_SVR_model")
        print(f"Modelo Linear SRV registrado no MLflow! Run ID: {run.info.run_id}")

def main():
    # Lendo o arquivo de dados
    # df = pd.read_csv('./data/dataset INCC.csv', sep="\t")
    df = pd.read_csv('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv', sep="\t")
    
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


if __name__ == "__main__":
    main() 