# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Importando as bibliotecas necessárias para análise
import pandas as pd
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from pandas_profiling import ProfileReport

# %%
os.chdir('C:/Users/mello/OneDrive/Documentos/10 - Projetos/Analise_Vendas_VideoGame_Kaggle/Base')

# %%
# Carregando o dataset
dados = pd.read_csv('vgsales.csv')
# %%
reporte=ProfileReport(dados, title='Pandas Profiling', explorative=True)

#%%
reporte.to_file("output/your_report.html")

#%%
reporte.to_widgets()

#%%
reporte.to_notebook_iframe()

# %%
dados.head(10)


# %%
# Renomeando as colunas
dados.columns = ['Ranking', 'Nome', 'Plataforma', 'Ano', 'Gênero', 'Editora', 'Vendas América do Norte', 'Vendas EUA', 'Vendas Japão', 'Outras Vendas', 'Vendas Globais']


# %%
# Após renomear as colunas, iremos visualizar as primeiras linhas do dataset
dados.head()


# %%
# Verificando as ultimas linhas do dataset
dados.tail()


# %%
# Verificando o tamanho do dataset
print('N. linhas:', dados.shape[0])
print('N. colunas:', dados.shape[1])


# %%
# Realizando uma estatística descritiva dos dados
dados.describe()


# %%
# Verificando os tipos das variáveis presentes no dataset
dados.dtypes


# %%
# Agora queremos saber qual a media de Vendas dos EUA
dados['Vendas EUA'].mean()


# %%
# Verificando qual a média de vendas globais
media_total_vendas = dados['Vendas Globais'].mean()
media_total_vendas


# %%
# Verificando qual a quantidade de Editoras no dataset
dados['Editora'].value_counts().head(15)


# %%
# Verificando a quantidade de Plataformas no dataset
dados['Plataforma'].value_counts().head(15)


# %%
# Agora iremos verificar qual foi o maior número de vendas do Japão
dados.sort_values(by='Vendas Japão',ascending = False)


# %%
# Agora queremos saber quais os valores unicos da Variavel Gênero
dados['Gênero'].unique()


# %%
# Verificando se existe alguma correlação entre as variáveis 
plt.figure(figsize=(14,9))
sns.heatmap(dados.corr(),annot=True,cmap='RdBu_r');

# %% [markdown]
# **Valores positivos mostram correlação positiva, enquanto valores negativos mostram correlação inversa. Como podemos ver, as vendas na América do Norte contribuíram consideravelmente para as vendas globais globais; portanto, podemos ver uma correlação positiva entre os dois.**

# %%
dados1 = dados.groupby(['Ano'])
plt.figure(figsize=(8,8))
dados1_mean = dados1['Vendas América do Norte','Vendas EUA','Vendas Japão','Outras Vendas'].aggregate(np.mean)
dados1_mean.plot(figsize=(18,7))
plt.title('Average sales over the course of years');

# %% [markdown]
# **Como podemos ver, a venda de videogames atingiu seu pico por volta de 1990, e foi diminuindo ao longo dos anos seguintes**

# %%
# Agora iremos analisar qual foi a frequencia de lançamento dos jogos ao longo do período
plt.rcParams['figure.figsize'] = (12,7)
sns.countplot(x=dados['Ano'],data = dados,palette="husl") 
plt.title("Frequência de lançamento do jogo")
plt.xlabel("Ano")
plt.ylabel("Frequencia")
plt.xticks(rotation=90)
plt.show()


# %%
# Share dos jogos por gênero

labels=dados.Gênero.value_counts().index
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = dados.Gênero.value_counts().values
# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette('Set2'), autopct='%1.1f%%')
plt.title('Jogos de acordo com o gênero',fontsize = 17,color = 'green');


# %%
# Top 15 das Plataformas que venderam mais
by_plataforma = dados.groupby('Plataforma')

by_plataforma['Vendas Globais'].sum().sort_values(ascending=False).head(15)


# %%
# Agora iremos plotar um gráfico para visualizar o TOP 15 das Plataformas que mais venderam 
dados_2 = dados.groupby('Plataforma')['Vendas Globais'].sum().sort_values(ascending = False).head(15).plot(kind='bar',
                                                                                                           figsize =(15,5),
                                                                                                           grid = False,
                                                                                                           rot = 0,
                                                                                                           color = ('lightseagreen'))
plt.title("TOP 15 das Plataformas que mais venderam")
plt.xlabel("Plataforma dos Jogos")
plt.ylabel("Vendas")
plt.show()


# %%
sns.lmplot(x="Vendas América do Norte", y="Vendas Globais", data=dados)
plt.show()


# %%
# TOP 10 Editoras que mais venderam
dados_2 = dados.groupby('Editora')['Vendas Globais'].sum().sort_values(ascending = False).head(10).plot(kind='bar',
                                                                                                           figsize =(15,5),
                                                                                                           grid = False,
                                                                                                           rot = 60,
                                                                                                           color = ('royalblue'))
plt.title("TOP 10 Editoras que mais venderam")
plt.xlabel("Plataforma dos Jogos")
plt.ylabel("Vendas")
plt.show()

# %% [markdown]
# **Percebemos que neste caso existe uma boa correlação entre as variáveis pois há uma Este tipo de correlação acontece quando há uma tendência crescente entre os pontos.**

# %%
plat_pop = pd.crosstab(dados.Plataforma,dados.Gênero)
plat_pop_total = plat_pop.sum(axis=1).sort_values(ascending= False)
plt.figure(figsize=(10, 10))
ax = sns.barplot(y = plat_pop_total.index, x = plat_pop_total.values)
ax.set_xlabel(xlabel ='Plataforma', fontsize= 15 )
ax.set_ylabel(ylabel ='Número de Jogos', fontsize= 15 )
ax.set_title(label='Número de jogos em cada plataforma', fontsize=15)
plt.show()


# %%
dados3 = dados.groupby(['Plataforma'])
val = dados3['Vendas América do Norte','Vendas EUA','Vendas Japão','Outras Vendas'].aggregate(np.mean)
plt.figure(figsize=(12,8))
ax = sns.boxplot(data=val, orient='h')
plt.xlabel('Receita por jogo',fontsize=16)
plt.ylabel('Região',fontsize=16)
plt.title('Distribuição de Vendas por Plataforma',fontsize=16);

# %% [markdown]
# **Após análise percebemos que há uma maior distribuição de vendas na Região da América do Norte**

# %%
# Número de jogos lançados pelas 20 maiores empresas de jogos
plt.figure(figsize=(15,6))
sns.barplot(x=dados.Editora.value_counts().index[:20],y=dados.Editora.value_counts().values[:20],data=dados)
plt.xticks(rotation=60)
plt.xlabel("Nome da empresa de jogos")
plt.ylabel("Número de jogos lançados")
plt.title("Número de jogos lançados pelas 20 maiores empresas de jogos",color="blue",fontsize=15);


