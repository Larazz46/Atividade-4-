import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Função para coletar dados históricos de preços de criptomoedas
def get_historical_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição à API: {e}")
        return None

# Coletando dados históricos do Bitcoin
coin_id = 'bitcoin'
days = 365
df = get_historical_data(coin_id, days)

if df is not None:
    print("Dados coletados com sucesso!")
    print(df.head())  # Exibe as primeiras linhas dos dados coletados

    # Criando a variável target (1 se o preço subiu, 0 se caiu)
    df['price_change'] = np.where(df['price'].shift(-1) > df['price'], 1, 0)

    # Removendo a última linha pois não temos o preço do próximo dia para ela
    df = df[:-1]

    # Dividindo os dados em features (X) e target (y)
    X = df[['price']]
    y = df['price_change']

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando um modelo de Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Fazendo previsões
    y_pred = model.predict(X_test)

    # Avaliando o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    # Visualizando a importância das variáveis
    importances = model.feature_importances_
    plt.figure(figsize=(8, 4))
    plt.bar(X.columns, importances)
    plt.title('Importância das Variáveis')
    plt.show()

    # Visualizando a distribuição das previsões
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Real')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Previsto')
    plt.title('Previsões vs Valores Reais (1 = Preço sobe, 0 = Preço cai)')
    plt.xlabel('Amostras')
    plt.ylabel('Previsão')
    plt.legend()
    plt.show()

    # Salvando os dados em um arquivo CSV
    df.to_csv(f'{coin_id}_historical_data.csv', index=False)
    print(f"\nDados salvos em '{coin_id}_historical_data.csv'.")
else:
    print("Não foi possível coletar os dados. Verifique sua conexão com a internet ou a API.")
