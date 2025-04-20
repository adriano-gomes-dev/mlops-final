import pandas as pd
from preprocessdata import process_data
import numpy as np
import requests
import evidently
from evidently import Report
from evidently import DataDefinition
from evidently.presets import RegressionPreset

def load_new_data():

    # X, y = process_data('./data/dataset INCC.csv')
    X, y = process_data('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv')
    return X, y

def get_predictions(data):
    print(data.head())

    # Defina as colunas esperadas pelo modelo
    columns = [ "Data" ]
    
    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in columns}
        instances.append(instance)


    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    
    response = requests.post(url, headers=headers, json=payload)
    predictions = response.json()
    predictions = predictions.get("predictions")
    print(predictions)
    return predictions

# Avaliar degradação do modelo
def evaluate_model(df, y):
    print("Avaliando modelo com dados originais")
    df["prediction"] = get_predictions(df)
    df["prediction"] = df["prediction"].astype(float)
    print(df["prediction"].unique())
    df["target"] = y
    print(df["target"].unique())
    report = Report(metrics=[RegressionPreset()])
    report.run(df, None)
    report.save_html("monitoring_report_df.html")
    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Score de drift: {drift_score}")
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
    print(f"Coluns drift: {drift_by_columns}")
    return drift_score, drift_by_columns

def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model(df_examples, y)

if __name__ == "__main__":
    main()