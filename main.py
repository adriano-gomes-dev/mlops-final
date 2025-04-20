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
from sklearn.model_selection import train_test_split

# Importando bibliotecas deste projeto
from core.preprocessdata import process_data
from core.trains import experiment_linear_regression, experiment_svr
from core.deployment import promote_model_to_production_based_on_mse

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

def main():
    # Caminho do dataset
    # datapath = '/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv'
    datapath = './data/dataset INCC.csv'

    # Preprocessamento dos dados para o modelo
    X, y = process_data(datapath)

    # MLFlow Conexão Info
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("INCC Tracking")

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

