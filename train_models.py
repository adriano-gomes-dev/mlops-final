#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análise de dados do INCC (Índice Nacional de Custo da Construção)
"""

# Importando bibliotecas necessárias
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# Importando bibliotecas deste projeto
from core.preprocessdata import process_data
from core.trains import experiment_linear_regression, experiment_svr
from core.deployment import promote_model_to_production_based_on_mse

def main():
    # MLFlow Conexão Info
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("INCC Tracking")
    
    df = process_data('./data/dataset INCC.csv')
    # df = process_data('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv')

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

    # Promove o melhor modelo
    promote_model_to_production_based_on_mse()

if __name__ == "__main__":
    main()