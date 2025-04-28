# Trabalho Final MLOps

Repositório do trabalho final em grupo da disciplina de MLOps - 2015/1 do curso de Pós-graduação lato sensu em nível de Especialização em Engenharia de Software para Aplicações de Ciência de Dados - UFRGS.

## Problema

Implementação do pipeline de uma aplicação de Machine Learning (usando práticas MLOps) utilizando o 
dataset os valores históricos do INCC (Indíce Nacional de Construção Civil) para previsão do crescimento 
futuro da taxa utilizando modelos de regressão linear.

## Estrutura

    core/               módulo contendo a implementação das etapas do pipeline
    data/               dataset utilizado
    monitor_model.py    monitoramento do modelo
    prediction_api.py   implementação da API
    train_models.py     script de execução do pipeline

## Requisitos mínimos

    pandas -> 2.0.0
    numpy -> 1.24.0
    matplotlib -> 3.7.0
    seaborn -> 0.12.0
    scipy -> 1.10.0
    scikit-learn -> 1.3.0
    plotly -> 5.18.0
    fastapi -> 0.100.0
    uvicorn -> 0.23.0
    mlflow -> 2.8.0
    evidently -> 0.6.7
    requests -> 2.31.0
    Python -> 3.10

## Modelos implementados

1. Regressão Linear
2. SVR
3. Random Forest Regressor

## Pipeline

1. Pré-processamento
2. Análise Exploratória
3. Implementação dos modelos
4. Deployment
5. Monitoramento

## Execução do pipeline

1. Inicialização do Mlflow com o comando:
    ```
    mlflow ui --backend-store-uri sqlite:///mlflow.d
    ```
2. Inicialização da API
    ```
    fastapi dev prediction_api.py
    ```
3. Iniciar treinamento inicial
    ```
    python train_models.py
    ```
4. Monitoramento
    ```
    python monitor_model.py
    ```

## Participantes

- Adriano Gebert Gomes
- Guilherme Mafra Soares
- Luigi Vaz Ferreira