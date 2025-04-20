import pandas as pd

def process_data(datapath):
    # Lendo o arquivo de dados
    df = pd.read_csv(datapath, sep="\t")
    
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

    # Preparando os dados para o modelo
    X = df[['Data']]
    y = df[['INCC Geral float']]

    return X, y