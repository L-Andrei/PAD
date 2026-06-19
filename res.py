import pandas as pd

# Carrega o arquivo com os dados
df = pd.read_csv('resultados_benchmark.csv')

# Calcula a média e a mediana de cada uma das coluna
resultados = pd.DataFrame({
    'Média': df.mean(),
    'Mediana': df.median()
}).round(2)

print(resultados)
