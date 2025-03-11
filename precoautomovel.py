import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Dados fictícios
data = {
    'Combustivel': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina'],
    'Idade': [5, 3, 2, 7, 4, 6, 1],
    'Quilometragem': [50000, 60000, 30000, 80000, 40000, 70000, 20000],
    'Preco': [30000, 25000, 35000, 20000, 28000, 22000, 40000]
}

df = pd.DataFrame(data)

# Dividindo os dados em features (X) e target (y)
X = df.drop('Preco', axis=1)
y = df['Preco']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo as transformações para cada tipo de dado
categorical_features = ['Combustivel']
numeric_features = ['Idade', 'Quilometragem']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Criando o pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse}')

# Visualizando as previsões vs valores reais
results = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
print("\nComparação entre Valores Reais e Previstos:")
print(results)

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valor Real')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Valor Previsto')
plt.title('Comparação entre Valores Reais e Previstos')
plt.xlabel('Amostras')
plt.ylabel('Preço')
plt.legend()
plt.show()
