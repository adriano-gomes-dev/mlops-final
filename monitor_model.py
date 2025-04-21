import pandas as pd
from preprocessdata import process_data
import numpy as np
import requests

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

def load_new_data():

    df = process_data('./data/dataset INCC.csv')
    # df = process_data('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv')
    df = df.sample(30)  # Pegamos exemplos aleatórios para testar

    # Preparando os dados para o modelo
    X = df[['Data']]
    y = df[['INCC Geral float']]

    return X, y.astype(float)

def get_predictions(data):
    print(data.head())

    # Defina as colunas esperadas pelo modelo
    columns = [ "Data" ]
    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in columns}
        instances.append(instance)

    url = "http://127.0.0.1:8000/items"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=instances)
    print(response)
    predictions = response.json()
    # predictions = predictions.get("predictions")
    print(predictions)
    return predictions

# Avaliar degradação do modelo
def evaluate_model(df, y):

    print("Avaliando modelo com dados originais")
    df["prediction"] = get_predictions(df)
    df["prediction"] = df["prediction"].astype(float)
    df["target"] = y

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=df, current_data=df)
    report.save_html("monitoring_report_df.html")

    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]

    print(f"Score de drift: {drift_score}")
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
    print(f"Coluns drift: {drift_by_columns}")
    return drift_score, drift_by_columns

def simulate_drift(df_examples):
    new_data = df_examples.copy()
    # Mudamos algumas colunas para simular mudanças nos padrões dos dados
    new_data["Data"] = new_data["Data"].apply(lambda x: x*-10)
    
    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        #os.system("python3 churn.py")
    else:
        if num_columns_drift >= 1:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            #os.system("python3 churn.py")
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")

def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model(df_examples, y)
    
    # Simular drift
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model(new_data, y)
    check_for_drift(drift_score, drift_by_columns)

if __name__ == "__main__":
    main()