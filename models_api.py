#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para API INCC (Índice Nacional de Custo da Construção)
"""

# Importando bibliotecas necessárias
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Importando bibliotecas deste projeto
from core.deployment import get_prediction_from_production_model

# API - FastAPI
# prescisa instalar o fastapi
# pip install fastapi[standard]
# e rodar com o comando: python -m fastapi dev main.py
from typing import List
import fastapi
from pydantic import BaseModel

class Item(BaseModel):
    Data: float

app = fastapi.FastAPI()

@app.post("/items/")
async def create_items(items: List[Item]):
    results = []
    for item in items:
        results.append(get_prediction_from_production_model(item.Data))
    return results

@app.get("/")
async def read_root():
    return {"message": "Predição de INCC. API para predição de INCC com base na data."}

@app.get("/predict/{date}")
async def predict(date: str):
    return {"date to predict": get_prediction_from_production_model(date)}

@app.get("/luigi/{var}")
async def luigi(var: str):
    return {"var": var}

