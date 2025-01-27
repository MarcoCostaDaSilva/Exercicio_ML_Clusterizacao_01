# Exercicio_ML_Clusterizacao_01

Este é o projeto prático do curso "Clusterização: Lidando com dados sem rótulos" da escola Alura.

**Enunciado:** Iremos desenvolver um projeto em uma empresa do setor de consultoria e marketing, com o objetivo de criar um modelo capaz de agrupar consumidores com base em seus interesses. Essa abordagem permitirá a elaboração de campanhas de marketing mais eficazes e direcionadas.

No decorrer do projeto foram utilizadas algumas bibliotecas Python no Google Colab.

- pandas 2.1.4
- matplotlib 3.7.1
- sklearn 1.3.2
- joblib 1.4.2
- numpy 1.26.4
- seaborn 0.13.1

Importando a base de dados:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns      

# URL da planilha compartilhada
url = 'https://docs.google.com/spreadsheets/d/1t7lpgUy45BSt6lBJWRCrBKFEltOsB0ojDjyZm4pBuYc/export?format=csv'

# Carregar a planilha como um DataFrame
df = pd.read_csv(url)
df

```
Conhecendo a estrutura de dados e iniciado o tratamento:
```python
df.info()

df['sexo'].unique()
```

Aplicando ENCODER:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories=[['F', 'M', 'NE']],sparse_output=False)

encoded_sexo = encoder.fit_transform(df[['sexo']])
encoded_sexo

encoded_df = pd.DataFrame(encoded_sexo,columns=encoder.get_feature_names_out(['sexo']))

dados = pd.concat([df,encoded_df],axis=1).drop('sexo',axis=1)
dados

import joblib

joblib.dump(encoder,'encoder.pkl')
```
Desenvolvendo o modelo:

```python
from sklearn.cluster import KMeans

mod_kmeans = KMeans(n_clusters=2, random_state=45)

modelo = mod_kmeans.fit(dados)
```
Avaliando o K-means

Inércia:

```python
mod_kmeans.inertia_
```

Silhueta:
```python
from sklearn.metrics import silhouette_score

silhouette_score(dados,mod_kmeans.predict(dados))
```

Avaliando métricas para diferentes K:
```python
def avaliacao(dados):
  inercia = []
  silhueta = []

  for k in range(2,21):
    kmeans = KMeans(n_clusters=k, random_state=45, n_init='auto')
    kmeans.fit(dados)
    inercia.append(kmeans.inertia_)
    silhueta.append(f'k={k} - '+ str(silhouette_score(dados, kmeans.predict(dados))))
  return silhueta, inercia
```

```python
silhueta, inercia = avaliacao(dados)
silhueta
```
Criamos uma função para visualizar o gráfico de silhueta para um determinado número de clusters. Este gráfico mostra como os dados são distribuídos dentro de cada cluster e quão bem eles são separados uns dos outros.

```python
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples

def graf_silhueta(n_clusters, dados):
  kmeans = KMeans(n_clusters=n_clusters, random_state=45, n_init='auto')
  cluster_previsoes = kmeans.fit_predict(dados)

  silhueta_media = silhouette_score(dados, cluster_previsoes)
  print(f'Valor médio para {n_clusters} clusters: {silhueta_media:.3f}')

  silhueta_amostra = silhouette_samples(dados, cluster_previsoes)

  fig, ax1 = plt.subplots(1, 1)
  fig.set_size_inches(9, 7)
  ax1.set_xlim([-0.1, 1])
  ax1.set_ylim([0, len(dados) + (n_clusters + 1) * 10])

  y_lower = 10
  for i in range(n_clusters):
      ith_cluster_silhueta_amostra = silhueta_amostra[cluster_previsoes == i]
      ith_cluster_silhueta_amostra.sort()

      tamanho_cluster_i = ith_cluster_silhueta_amostra.shape[0]
      y_upper = y_lower + tamanho_cluster_i

      cor = cm.nipy_spectral(float(i) / n_clusters)
      ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhueta_amostra,
                        facecolor=cor, edgecolor=cor, alpha=0.7)

      ax1.text(-0.05, y_lower + 0.5 * tamanho_cluster_i, str(i))
      y_lower = y_upper + 10

  ax1.axvline(x=silhueta_media, color='red', linestyle='--')
  ax1.set_title(f'Gráfico da Silhueta para {n_clusters} clusters')
  ax1.set_xlabel('Valores do coeficiente de silhueta')
  ax1.set_ylabel('Rótulo do cluster')

  ax1.set_yticks([])
  ax1.set_xticks([i / 10.0 for i in range(-1, 11)])
  plt.show()

graf_silhueta(2, dados)
```
![image](https://github.com/user-attachments/assets/c0d43c64-7701-45df-87d7-8a6170c1b8b0)

Definimos e executamos a função plot_cotovelo para criar um gráfico do método do cotovelo, que ajuda a identificar o número ótimo de clusters observando onde a curva de inércia começa a se estabilizar.

```python
def plot_cotovelo(inercia):
  plt.figure(figsize=(8,4))
  plt.plot(range(2,21),inercia,'bo-')
  plt.xlabel('Número de clusters')
  plt.ylabel('Inércia')
  plt.title('Método do Cotovelo para Determinação de k')
  plt.show()

```
![image](https://github.com/user-attachments/assets/1d5135e1-623f-480f-8acf-5d45c6af4df8)


Avaliação e ajustes de dados:

Primeiramente, utilizamos o método .describe() para exibir estatísticas resumidas que ajudam a entender a distribuição de valores de cada coluna no DataFrame dados.

```python
dados.describe()
```

Aplicação de normalização:

Em seguida, aplicamos uma transformação de escala aos dados utilizando o MinMaxScaler do scikit-learn, que redimensiona os recursos para um intervalo dado, normalmente 0 a 1, para garantir que a escala dos dados não distorça o desempenho dos algoritmos de aprendizado de máquina.
```python

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

dados_escalados = scaler.fit_transform(dados)

dados_escalados
```
Convertendo os dados escalados de volta para um DataFrame pandas para facilitar a análise e a visualização, preservando os nomes das colunas originais.
Salvamos o objeto scaler utilizando joblib, permitindo que o mesmo scaler seja reutilizado em novos dados no futuro, mantendo a consistência na escala aplicada.
```python
dados_escalados = pd.DataFrame(dados_escalados, columns=dados.columns)

dados_escalados.describe()

joblib.dump(scaler, 'scaler.pkl')
```
Verificando as métricas para os novos dados:
Chamamos a função avaliacao anteriormente definida para calcular a inércia e o silhouette score de diferentes configurações de clusters utilizando os dados escalados.
```python
silhueta, inercia = avaliacao(dados_escalados)
silhueta
```
Com o resultado, escolhemos o valor de k = 3 como o melhor. Geramos o gráfico de silhueta para 3 clusters com os dados escalados, visualizando como os dados são distribuídos dentro dos clusters e quão bem separados eles estão.

```python
graf_silhueta(3, dados_escalados)
```
![image](https://github.com/user-attachments/assets/92aeef95-9f0a-481a-9d8f-06197189d0b2)

Plotagem do método cotovelo:

```python
plot_cotovelo(inercia)
```
![image](https://github.com/user-attachments/assets/97c2c42a-25e3-4482-997b-df8a2c423616)

Com o método do cotovelo, concluímos que o modelo K-means com 3 agrupamentos se sai melhor. Treinamos um modelo KMeans com 3 clusters usando os dados escalados. A especificação de random_state e n_init garante a reprodutibilidade e a estabilidade na inicialização dos clusters.

```python
modelo_kmeans = KMeans(n_clusters=3, random_state=45, n_init='auto')
modelo_kmeans.fit(dados_escalados)
```
Finalmente, salvamos o modelo treinado utilizando joblib para que possa ser reutilizado ou implementado em produção sem a necessidade de re-treinamento.

```python
joblib.dump(modelo_kmeans, 'kmeans.pkl')
```
Primeiro, criamos um novo DataFrame para analisar os dados originalmente escalados, revertendo a escala para os valores originais usando o MinMaxScaler previamente treinado. Isso permite uma análise mais intuitiva dos dados em sua escala original.

```python
dados_analise = pd.DataFrame()
dados_analise[dados_escalados.columns] = scaler.inverse_transform(dados_escalados)
dados_analise
```
Adicionamos ao DataFrame dados_analise uma nova coluna chamada 'cluster', que contém os rótulos dos clusters atribuídos a cada ponto de dado pelo modelo KMeans. Em seguida, calculamos as médias dos atributos para cada cluster para identificar características predominantes em cada grupo.

```python
dados_analise['cluster'] = modelo_kmeans.labels_
cluster_media = dados_analise.groupby('cluster').mean()
cluster_media.T
```
Transpomos o DataFrame cluster_media para que os clusters se tornem colunas e os atributos se tornem linhas. Isso facilita a visualização e comparação das médias dos atributos entre os clusters.

```python
cluster_media = cluster_media.transpose()
cluster_media.columns = [0, 1, 2]
```
Exibimos as médias dos atributos para cada cluster em ordem decrescente. Essa visualização ajuda a entender quais atributos são mais significativos em cada cluster, facilitando a interpretação de quais características definem cada grupo.

```python
cluster_media[0].sort_values(ascending=False)
cluster_media[1].sort_values(ascending=False)
cluster_media[2].sort_values(ascending=False)
```
Ao fim, conseguimos entender as características dos 3 grupos e traçar estratégias de marketing que podem ser eficientes. Podemos resumir essas estratégias da seguinte forma:

Grupo 0 é focado em um público jovem com forte interesse em moda, música e aparência.
Grupo 1 está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
Grupo 2 é mais equilibrado, com interesses em música, dança e moda.

