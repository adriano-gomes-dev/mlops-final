import logging
import requests
import os
from datetime import datetime, timedelta

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# Importando bibliotecas deste projeto
from core.preprocessdata import process_data

import warnings
from sklearn.exceptions import UndefinedMetricWarning

log_name = 'date.log'

logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def load_new_data():

    df = process_data('./data/dataset INCC.csv')
    # df = process_data('/Users/gms/MLOPS/mlops-final/data/dataset INCC.csv')
    df = df.sample(30)  # Pegamos exemplos aleatórios para testar

    # Preparando os dados para o modelo
    X = df[['Data']]
    y = df[['INCC Geral float']]

    return X, y.astype(float)

def get_predictions(data):
    #print(data.head())

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
    return predictions

# Avaliar degradação do modelo
def evaluate_model(df, y, new_data=None):

    if new_data is None:
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
        return drift_score, drift_by_columns
    else:
        print("Avaliando modelo com dados novos")
        new_data["prediction"] = get_predictions(new_data)
        new_data["prediction"] = new_data["prediction"].astype(float)
        new_data["target"] = y

        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=df, current_data=new_data)
        report.save_html("monitoring_report_new_data.html")

        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]

        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        return drift_score, drift_by_columns

def retrain_model(message):
    print("Treinando novo modelo...")
    os.system("python3 train_models.py")
    logger.info(f"Re-trained|{message}|{datetime.today().strftime('%d/%m/%Y')}")
    print("Treino finalizado!")

def simulate_drift(df_examples):
    new_data = df_examples.copy()
    # Mudamos algumas colunas para simular mudanças nos padrões dos dados
    new_data["Data"] = new_data["Data"].apply(lambda x: x+100)
    
    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        retrain_model("Drift detectado no Dataset")
    else:
        if num_columns_drift >= 1:
            retrain_model(f"Drift detectado em {num_columns_drift} colunas")
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")

def check_last_train_date():
    current_date = datetime.today()
    try:
        with open(log_name) as log:
            lines = log.readlines()
            last_log = lines[0]
        log_date_str = last_log.split('|')[-1]
        log_date = datetime.strptime(log_date_str, '%d/%m/%Y\n')
        if (log_date + timedelta(days= 365)) < current_date:
            retrain_model("Intervalo de 1 ano")
    except:
        retrain_model("Não identificado último log de treino")


def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model(df_examples, y)
    
    # Simular drift
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)
    check_for_drift(drift_score, drift_by_columns)
    check_last_train_date()

if __name__ == "__main__":
    main()