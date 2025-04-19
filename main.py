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

import fastapi

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://localhost:5000")

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/predict/{date}")
def predict(date: str):
    return {"date to predict": date}


def experiment_linear_regression(df):

    # Preparando os dados para o modelo
    X = df[['Data']]
    y = df[['INCC Geral float']]

    # Dividindo os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nDimensões dos conjuntos de dados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}") 
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

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


def main():
    # Lendo o arquivo de dados
    df = pd.read_csv('./data/dataset INCC.csv', sep="\t")
    
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

    mlflow.set_experiment("INCC Tracking")

    # Experimento 1: Regressão Linear
    experiment_linear_regression(df)


if __name__ == "__main__":
    main() 