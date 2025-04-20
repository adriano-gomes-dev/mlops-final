#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para API INCC (Índice Nacional de Custo da Construção)
"""

# Importando bibliotecas necessárias
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

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

def get_prediction_from_production_model(value_to_predict):
    # Set the tracking URI to the local MLflow server
    # precisa chamar aqui novamente por causa do fastapi
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    model_production = None

    for mv in client.search_model_versions("name='incc_model'"):
        print(f"Version: {mv.version}, Stage: {mv.current_stage}")
        
        if mv.current_stage == "Production":
            model_production = mlflow.pyfunc.load_model(model_uri=f"models:/{mv.name}/{mv.version}")
            break
    
    data_to_predict = pd.DataFrame({'Data': [value_to_predict]})
    predictions = model_production.predict(data_to_predict)
    return str(predictions[0][0])
