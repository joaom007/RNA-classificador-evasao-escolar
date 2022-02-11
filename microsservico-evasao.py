# -*- coding: utf-8 -*-
"""
Criado Jul 2021
Autor: João Marcos, 
github: https://github.com/joaom007
linkedin: https://www.linkedin.com/in/joaomarcos17/

Trabalho de conclusão de curso apresentado à Coordenadoria do Curso de Pós
Graduação em Desenvolvimento Web do câmpus Itapetininga do Instituto Federal 
de São Paulo para a obtenção do título de Especialista em Desenvolvimento Web.

Tema: Aprendizado de Máquina e a Evasão Escolar - Microsserviço para análise 
e predição de evasão no IFSP câmpus Itapetininga.
"""

import pandas as pd

"""### ETAPA_1: Aquisição de dados do csv e visualização básica"""

base_1 = pd.read_csv('!dados-IFSP-ML-TCC.csv', encoding='utf-8-sig')

base_1.head()
base_1.info()
base_1.describe()
base_1.columns
base_1.count
base_1.dtypes

import numpy as np

"""### ETAPA 2: Configurações na estrutura dos dados"""

#substitui '-' por nan (Not a Number)
base_2 = base_1.replace(['-'], np.NaN)
base_2 = base_2.replace([''], np.NaN)

#converte virgula para ponto em colunas necessárias
base_2['Frequência no Período'] = base_2['Frequência no Período'].str.replace(',', '.')
base_2['I.R.A.'] = base_2['I.R.A.'].str.replace(',', '.')
base_2['Percentual de Progresso'] = base_2['Percentual de Progresso'].str.replace(',', '.')
base_2['Renda Bruta Familiar (R$)'] = base_2['Renda Bruta Familiar (R$)'].str.replace('.', '')
base_2['Renda Bruta Familiar (R$)'] = base_2['Renda Bruta Familiar (R$)'].str.replace(',', '.')
base_2['Renda Per Capita'] = base_2['Renda Per Capita'].str.replace('.', '')
base_2['Renda Per Capita'] = base_2['Renda Per Capita'].str.replace(',', '.')
base_2['Endereço'] = base_2['Endereço'].str.replace('nan', '')

#converte tipo do atributo de genéricos para numérico, datas e strings
base_2['Ano de Conclusão do Ensino Anterior'] = pd.to_numeric(
    base_2['Ano de Conclusão do Ensino Anterior'])
base_2['Frequência no Período'] = pd.to_numeric(base_2['Frequência no Período'])
base_2['Município de Residência (Código IBGE)'] = pd.to_numeric(
    base_2['Município de Residência (Código IBGE)'])
base_2['I.R.A.'] = pd.to_numeric(base_2['I.R.A.'])
base_2['Percentual de Progresso'] = pd.to_numeric(base_2['Percentual de Progresso'])
base_2['Renda Bruta Familiar (R$)'] = pd.to_numeric(base_2['Renda Bruta Familiar (R$)'])
base_2['Renda Per Capita'] = pd.to_numeric(base_2['Renda Per Capita'])
base_2['Data de Matrícula'] = pd.to_datetime(base_2['Data de Matrícula'])
base_2['Data de Nascimento'] = pd.to_datetime(base_2['Data de Nascimento'])
base_2['Estado Civil'] = base_2['Estado Civil'].astype(str)
base_2['Etnia/Raça'] = base_2['Etnia/Raça'].astype(str)
base_2['Endereço'] = base_2['Endereço'].astype(str)

base_2 = base_2.replace(['nan'], np.NaN)

#Verificação valores faltantes por índice
base_2.isnull().sum()


#Remoção de registros de cursos de extensão
filtro = base_2['Modalidade'] == 'FIC'
cursos_extensao = base_2.loc[filtro]
base_2.drop(cursos_extensao.index, inplace=True)

#remoção de registros de matriculas após 2020
base_2['Data de Matrícula'] = base_2['Data de Matrícula'].apply(lambda x: x.year)
filtro = base_2['Data de Matrícula'] >= 2020
apos_pandemia = base_2.loc[filtro]
base_2.drop(apos_pandemia.index, inplace=True)
base_2.reset_index(drop=True, inplace=True)

#exclusão de colunas com dados nulos
base_2 = base_2.drop(columns=['Data da Colação'])
base_2 = base_2.drop(columns=['Período Letivo de Integralização']) 
base_2 = base_2.drop(columns=['Polo'])                       
#75% de dados nulos
base_2 = base_2.drop(columns=['Data de Conclusão de Curso']) 
base_2 = base_2.drop(columns=['Renda Bruta Familiar (R$)'])
base_2 = base_2.drop(columns=['Renda Per Capita'])
base_2 = base_2.drop(columns=['Estado Civil'])

#verificando colunas com todos os dados iguais
np.unique(base_2['Campus'], return_counts=True) #Único valor = ITP
base_2 = base_2.drop(columns=['Campus'])

#exclusão de colunas com dados redundantes
base_2 = base_2.drop(columns=['Data de Integralização'])
base_2 = base_2.drop(columns=['Ano de Ingresso'])
base_2 = base_2.drop(columns=['Município de Residência (Código IBGE)'])
base_2 = base_2.drop(columns=['Município de Residência'])
base_2 = base_2.drop(columns=['Descrição do Curso'])
base_2 = base_2.drop(columns=['Modalidade'])
base_2 = base_2.drop(columns=['Situação no Período'])

#exclusão de colunas nominais (classificação de grupos)
base_2 = base_2.drop(columns=['Turma'])

#verifica valores faltantes por indice
base_2.isnull().sum()



"""### ETAPA_3 resolvendo valores faltantes"""
#renomeando indice data de matrícula para ano de matrícula
base_3 = base_2.rename(columns={'Data de Matrícula': 'Ano de Matrícula'})

#preenchendo ano anterior com a moda da diferença de anos
diferenca_ingresso = base_3['Ano de Matrícula'] - base_3['Ano de Conclusão do Ensino Anterior']
diferenca_ingresso.describe()
moda = diferenca_ingresso.mode()
moda_faltante = base_3['Ano de Matrícula'] - int(moda)
#substitui anos faltantes pela moda
base_3['Ano de Conclusão do Ensino Anterior'] = base_3[
    'Ano de Conclusão do Ensino Anterior'].fillna(moda_faltante)

#substituindo cidades faltantes com a mais frequênte
# summary, repeat = np.unique(base_3['Cidade'], return_counts=True)
base_3['Cidade'] = base_3['Cidade'].fillna('ITAPETININGA')

#substituindo tipo de escola faltante com a mais frequênte
# summary, repeat = np.unique(base_3['Tipo de Escola de Origem'], return_counts=True)
base_3['Tipo de Escola de Origem'] = base_3['Tipo de Escola de Origem'].fillna('Pública')

#Gera arquivo csv já pré-processado
# base_3.to_csv('!base3-preprocessamento.csv', index=False, encoding='utf-8-sig')
#verifica valores faltantes por indice  
base_3.isnull().sum()


from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import googlemaps

"""### ETAPA_4 Resolvendo inconsistências dos atributos"""

# base_4 = pd.read_csv('!base3-preprocessamento.csv', encoding='utf-8-sig')
base_4 = base_3.copy()
base_4.dtypes

#Convertendo numéricos inteiros para tipo int
base_4['Período Atual'] = base_4['Período Atual'].apply(lambda x: int(x))
base_4['Ano de Conclusão do Ensino Anterior'] = base_4[
    'Ano de Conclusão do Ensino Anterior'].apply(lambda x: int(x))

# Convertendo data de nascimento em idade
def idade(nascimento):
    hoje = date.today()
    return hoje.year - nascimento.year - (
        (hoje.month, hoje.day) < (nascimento.month, nascimento.day))

base_4['Data de Nascimento'] = base_4['Data de Nascimento'].apply(idade)

#renomeando índice data de nascimento para idade
base_4 = base_4.rename(columns={'Data de Nascimento': 'Idade'})
summary, repeat = np.unique(base_4['Idade'], return_counts=True)
sns.set_theme(style="darkgrid")
graf_1 = sns.countplot(x = base_4['Idade'])

#legenda dos eixos
plt.ylabel('Frequência', fontsize=16)
plt.xlabel('Idade', fontsize=16)
plt.show()


filtro = base_4['Idade'] < 16
idades_incorretas = base_4.loc[filtro]
base_4.drop(idades_incorretas.index, inplace=True)
base_4.reset_index(drop=True, inplace=True)

#visualizando novamente
graf_1 = sns.countplot(x = base_4['Idade'])

#legenda dos eixos
plt.ylabel('Frequência', fontsize=16)
plt.xlabel('Idade', fontsize=16)
plt.show()


#Calculando diferença em anos entre ano de matrícula e ano de conclusão do ensino anterior
diferenca_ingresso = base_4['Ano de Matrícula'] - base_4['Ano de Conclusão do Ensino Anterior']
#visualizando em um histograma
graf_2 = sns.countplot(x = diferenca_ingresso)
#legenda dos eixos
plt.ylabel('Frequência', fontsize=16)
plt.xlabel('Ano(s)', fontsize=16)
# plt.title('Diferença em anos entre matrícula e ensino anterior', fontsize=20)
plt.show()

#Capturando registros com ano anterior após ano de mátricula
filtro = diferenca_ingresso < 0
diferenca = base_4.loc[filtro]



#Ajustar endereço
base_4['Endereço'] = base_4['Endereço'].str.replace(',,', ',')
base_4['Endereço'] = base_4['Endereço'].str.replace(', ,', ',')

#criando nova base com endereços para exportar em csv
# base_endereco = pd.DataFrame(
#     base_4['Endereço'], columns=['Endereço', 'Distância do Câmpus (Km)'])

#chave da API google
gmaps = googlemaps.Client(key='')
campus = 'Av. João Olímpio de Oliveira, 1561 - Vila Asem, Itapetininga - SP, 18202-000';
#Função para consumir API google e obter disntância entre câmpus e endereço
def obter_distancia_km(endereco):
    try:
        consulta = gmaps.distance_matrix(campus, endereco, mode='driving', units='metric')
        return consulta['rows'][0]['elements'][0]['distance']['value']/1000
    except (KeyError, ValueError, TypeError):
        print('Falha na conversão do endereço: {}'. format(endereco))
        return endereco
    
# base_endereco['Distância do Câmpus (Km)'] = base_endereco['Endereço'].apply(obter_distancia_km)

#exportar csv com endereço e distância
# base_endereco.to_csv('!base-endereco-preprocessamento.csv', index=True, encoding='utf-8-sig')
#ler csv com endereço e distância
# base_endereco = pd.read_csv('!base-endereco-preprocessamento.csv', encoding='utf-8-sig', index_col=0)
#buscando endereços não tratados
# filtro = base_endereco['Distância do Câmpus (Km)'].apply(lambda x: len(x) > 10)
# enderecos_converter = base_4['Endereço'].loc[filtro]

# enderecos_converter = enderecos_converter.str.replace('KM 158, 158', ' 158')
# divisao = enderecos_converter.str.split(', ')
# ceps = divisao.str.get(3)
# ceps = ceps.apply(obter_distancia_km)
# ceps[528] = 28.4
# ceps[913] = 47.7
# ceps[1041] = 4.4
# base_endereco['Distância do Câmpus (Km)'].loc[filtro] = ceps

#exportar csv com endereço e distância
# base_endereco.to_csv('!base-endereco-completa-preprocessamento.csv', index=True, encoding='utf-8-sig')

# base_endereco = pd.read_csv('!base-endereco-completa-preprocessamento.csv', encoding='utf-8-sig', index_col=0)

# base_4 = base_4.rename(columns={'Endereço': 'Distância do Câmpus (Km)'})
# base_4['Distância do Câmpus (Km)'] = base_endereco['Distância do Câmpus (Km)']

# base_4.to_csv('!base4-preprocessamento.csv', index=False, encoding='utf-8-sig')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""### milestone 1, iniciar aqui base já possui algum pré-processamento"""

base_4 = pd.read_csv('!base4-preprocessamento.csv', encoding='utf-8-sig')

base_4 = base_4.drop(columns=['Cidade'])

filtro = base_4['Distância do Câmpus (Km)'] > 200
altas_distancias = base_4.loc[filtro]
base_4.drop(altas_distancias.index, inplace=True)
base_4.reset_index(drop=True, inplace=True)

def without_hue(plot, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() * 0.5
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12, ha='center')
    plt.show()
    
#Verificando valores únicos
sns.set_style('darkgrid')

# ax = sns.countplot(x = base_4['Nível de Ensino'], hue = base_4['Nível de Ensino'], dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Nível de Ensino', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Nível de Ensino'])

def plotar_situacao_curso(ax, dataset):
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.xticks(size = 10)
    plt.xlabel('Situação no Curso', size = 12)
    plt.yticks(size = 10)
    plt.ylabel('Aluno(s)', size = 12)
    without_hue(ax, dataset['Situação no Curso'])

# ax = sns.countplot(x = base_4['Situação no Curso'], hue = base_4['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, base_4)

np.unique(base_4['Situação no Curso'])
filtro = base_4['Situação no Curso'] == 'Cancelado'
filtro2 = base_4['Situação no Curso'] == 'Cancelamento Compulsório'
filtro3 = base_4['Situação no Curso'] == 'Cancelamento por Desligamento'
filtro4 = base_4['Situação no Curso'] == 'Jubilado'
filtro5 = base_4['Situação no Curso'] == 'Transferido Externo'
filtro6 = base_4['Situação no Curso'] == 'Transferido Interno'
base_4['Situação no Curso'].loc[filtro] = 'Evasão'
base_4['Situação no Curso'].loc[filtro2] = 'Evasão'
base_4['Situação no Curso'].loc[filtro3] = 'Evasão'
base_4['Situação no Curso'].loc[filtro4] = 'Evasão'
base_4['Situação no Curso'].loc[filtro5] = 'Evasão'
base_4['Situação no Curso'].loc[filtro6] = 'Evasão'

# ax = sns.countplot(x = base_4['Situação no Curso'], hue = base_4['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, base_4)


# base_4['Período Atual']
# ax = sns.countplot(x = base_4['Período Atual'], hue = base_4['Período Atual'], dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Período Atual', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Período Atual'])


# #situação do curso para o 1º periodo
# # filtro =  base_4['Período Atual'] == 1
# filtro =  base_4['Período Atual'] == 4
# primeiro_periodo = base_4.loc[filtro]
# ax = sns.countplot(x = primeiro_periodo['Situação no Curso'], hue = primeiro_periodo['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, primeiro_periodo)


# ax = sns.countplot(x = base_4['Turno'], hue = base_4['Turno'], dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Turno', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Turno'])


# # ax = sns.countplot(x = base_4['Sexo'], hue = base_4['Sexo'], dodge=False)
# # plt.xticks(size = 10)
# # plt.xlabel('Sexo', size = 12)
# # plt.yticks(size = 10)
# # plt.ylabel('Aluno(s)', size = 12)
# # without_hue(ax, base_4['Sexo'])

# !pip install plotly --upgrade
# pip install -U kaleido

import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# # pio.renderers.default='png'
# colors = base_4['Situação no Curso'].map(
#     {'Evasão': 0, 'Concluído': 1, 'Formado': 2,
#       'Matriculado': 3, 'Matrícula Vínculo Institucional': 4,
#       'Trancado': 5  })
# graf_5 = px.parallel_categories(base_4, dimensions=['Turno', 'Situação no Curso', 'Sexo'], color=colors)
# graf_5.update_layout(autosize=True, margin=dict(l=100), font=dict(size=20))
# graf_5.show()

# #situação do curso por sexo
# filtro =  base_4['Sexo'] == 'M'
# masculino = base_4.loc[filtro]
# ax = sns.countplot(x = masculino['Situação no Curso'], hue = masculino['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, masculino)

# filtro = base_4['Sexo'] == 'F'
# feminino = base_4.loc[filtro]
# ax = sns.countplot(x = feminino['Situação no Curso'], hue = feminino['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, feminino)

# filtro = base_4['Turno'] == 'Noturno'
# noturno = base_4.loc[filtro]
# ax = sns.countplot(x = noturno['Situação no Curso'], hue = noturno['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, noturno)

# filtro = base_4['Turno'] == 'Vespertino'
# vespertino = base_4.loc[filtro]
# ax = sns.countplot(x = vespertino['Situação no Curso'], hue = vespertino['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, vespertino)

# filtro = base_4['Turno'] == 'Matutino'
# matutino = base_4.loc[filtro]
# ax = sns.countplot(x = matutino['Situação no Curso'], hue = matutino['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, matutino)

# filtro = base_4['Turno'] == 'Integral'
# integral = base_4.loc[filtro]
# ax = sns.countplot(x = integral['Situação no Curso'], hue = integral['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, integral)

# filtro = base_4['Código Curso'] == 'ITP.TEC.MSI.2010'
# tec_msi = base_4.loc[filtro]


# ax = sns.countplot(x = base_4['Código Curso'], hue = base_4['Código Curso'], dodge=False, order = base_4['Código Curso'].value_counts().index)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.xticks(size = 10)
# plt.xlabel('Código Curso', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Código Curso'])

np.unique(base_4['Código Curso'])
filtro = base_4['Código Curso'] == 'ITP.LIC.FIS.2010.'
filtro2 = base_4['Código Curso'] == 'ITP.LIC.FIS.2010..'
base_4['Código Curso'].loc[filtro] = 'ITP.LIC.FIS.2010'
base_4['Código Curso'].loc[filtro2] = 'ITP.LIC.FIS.2010'


# #visualização apenas da situação de evasão do curso
# filtro = base_4['Situação no Curso'] == 'Evasão'
# evadidos_curso = base_4.loc[filtro]
# ax = sns.countplot(
#     x = evadidos_curso['Código Curso'],
#     hue = evadidos_curso['Código Curso'],
#     dodge=False,
#     order = evadidos_curso['Código Curso'].value_counts().index)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.xticks(size = 10)
# plt.xlabel('Taxa de Evasão por Curso', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, evadidos_curso['Código Curso'])



# ax = sns.countplot(x = base_4['Etnia/Raça'], hue = base_4['Etnia/Raça'], dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Etnia/Raça', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Etnia/Raça'])

# ax = sns.countplot(x = base_4['Tipo de Escola de Origem'], hue = base_4['Tipo de Escola de Origem'], dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Tipo de Escola de Origem', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, base_4['Tipo de Escola de Origem'])

# colors = base_4['Situação no Curso'].map(
#     {'Evasão': 0, 'Concluído': 1, 'Formado': 2,
#       'Matriculado': 3, 'Matrícula Vínculo Institucional': 4,
#       'Trancado': 5  })
# graf_5 = px.parallel_categories(base_4, dimensions=['Código Curso', 'Situação no Curso', 'Tipo de Escola de Origem'], color=colors)
# graf_5.update_layout(autosize=True, margin=dict(l=150, r=150), font=dict(size=20))
# graf_5.show()


# #visualização apenas da situação de evasão por tipo de escola de origem
# filtro = base_4['Tipo de Escola de Origem'] == 'Pública'
# escola_publica = base_4.loc[filtro]
# ax = sns.countplot(x = escola_publica['Situação no Curso'], hue = escola_publica['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, escola_publica)

# filtro = base_4['Tipo de Escola de Origem'] == 'Privada'
# escola_privada = base_4.loc[filtro]
# ax = sns.countplot(x = escola_privada['Situação no Curso'], hue = escola_privada['Situação no Curso'], dodge=False)
# plotar_situacao_curso(ax, escola_privada)

#ajuste de tempo de curso negativo
tempo_curso = base_4['Ano Letivo de Previsão de Conclusão'] - base_4['Ano de Matrícula']
filtro = tempo_curso < 0
tempo_erro = base_4.loc[filtro]
base_4.drop(tempo_erro.index, inplace=True)
base_4.reset_index(drop=True, inplace=True)
tempo_curso = base_4['Ano Letivo de Previsão de Conclusão'] - base_4['Ano de Matrícula']


# np.unique(tempo_curso, return_counts=True)
# ax = sns.countplot(x = tempo_curso, hue = tempo_curso, dodge=False)
# plt.xticks(size = 10)
# plt.xlabel('Tempo de curso em ano(s)', size = 12)
# plt.yticks(size = 10)
# plt.ylabel('Aluno(s)', size = 12)
# without_hue(ax, tempo_curso)


# filtro = tempo_curso == 5
# tempo_1ano = base_4.loc[filtro]


# colors = base_4['Sexo'].map({'M': 0, 'F': 1})
# graf_5 = px.parallel_categories(base_4, dimensions=['Nível de Ensino', 'Sexo', 'Ano de Matrícula' ], color=colors)
# graf_5.update_layout(autosize=True, margin=dict(l=100, r=100), font=dict(size=20))
# graf_5.show()

# colors = base_4['Sexo'].map({'M': 0, 'F': 1})
# graf_5 = px.parallel_categories(base_4, dimensions=['Turno', 'Sexo', 'Ano de Matrícula'], color=colors)
# graf_5.update_layout(autosize=True, margin=dict(l=100, r=100), font=dict(size=20))
# graf_5.show()



# ax = px.scatter_matrix(base_4, dimensions=[
#     'Percentual de Progresso', 'I.R.A.', 'Distância do Câmpus (Km)'], color='Situação no Curso')
# ax.show()


# ax = px.scatter_matrix(base_4, dimensions=[
#     'Frequência no Período', 'I.R.A.', 'Idade'], color='Situação no Curso')
# ax.show()

def calculaPercentil98(df):

    # Define os percentis para ver como os dados estão distribuídos
    description = df.describe(percentiles=[.25, .5, .75, .9, .95, .98, .99, .999])
    description_dict = description.to_dict()

    # Coleta o valor que delimita 98% das amostras no DataFrame
    data_cut = float(description_dict["Distância do Câmpus (Km)"]["98%"])

    # Seleciona apenas as amostras dentro do intervalo de 98%, removendo os outliers
    df98 = df[df["Distância do Câmpus (Km)"] <= data_cut]

    return df98


df98 = calculaPercentil98(evadidos_curso)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Criando o ambiente do gráfico 
sns.set_style("white")
plt.figure(figsize=(15, 10))

# Gráfico de Dispersão
cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)
g = sns.scatterplot(x="Situação no Curso", y="Distância do Câmpus (Km)", 
                    hue="Distância do Câmpus (Km)", size="Distância do Câmpus (Km)",
                    palette=cmap, data=df98) #df98 calculado anteriormente

# Ajusta rótulos
g.set_title("Distância do Câmpus (Km)")
g.set_xlabel("Situação no Curso")
g.yaxis.set_major_locator(ticker.MultipleLocator(1))

for ind, label in enumerate(g.get_xticklabels()):
    if ind % 4 == 0:  # Mantém apenas os rótulos múltiplos de 4 no eixo x
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

bins = 20

# Criando o ambiente do gráfico 
sns.set_style("white")
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Insere curva KDE (Kernel Density Estimation)
g1 = sns.distplot(df98["Distância do Câmpus (Km)"], ax=ax, 
                  kde=True, hist=False) 

# Insere histograma
ax_copy = ax.twinx()
g2 = sns.distplot(df98["Distância do Câmpus (Km)"], ax=ax_copy, kde=False, hist=True, 
             bins=bins, norm_hist=False)

# Ajusta rótulos
g1.set_ylabel("Probabilidade")
g2.set_ylabel("Qauantidade")
g2.set_title("BTC BTG Pactual - " + datetime.now().strftime("%Y-%m-%d %H:%M"))
g1.xaxis.set_major_locator(ticker.MultipleLocator((df98["Distância do Câmpus (Km)"].max()-df98["Distância do Câmpus (Km)"].min())/bins))
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()






# #visualização apenas da situação de evasão por idade, 
# sns.set_theme(style="darkgrid")
# filtro = base_4['Situação no Curso'] == 'Evasão'
# evadidos_curso = base_4.loc[filtro]
# graf_1 = sns.countplot(x = evadidos_curso['Idade'])
# plt.ylabel('Frequência', fontsize=16)
# plt.xlabel('Idade', fontsize=16)
# plt.show()



base_4.columns


# base_4.to_csv('!base4-1-preprocessamento.csv', index=False, encoding='utf-8-sig')


"""### milestone 2 pode-se começar por aqui com o pré-processamento já realizado"""
import pandas as pd
import seaborn as sns
base_4 = pd.read_csv('!base4-1-preprocessamento.csv', encoding='utf-8-sig')

#evasão = True
base_4['Situação no Curso'] = base_4['Situação no Curso'] == 'Evasão'


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pickle

"""### ETAPA_5 Transformação de atributos categóricos em numéricos"""

base_5 = base_4.drop(columns=['Situação no Curso'])

#Definindo campos previsores: de 0 à 15 na base_5
previsores = base_5.iloc[:, 0:15].values
#Definindo campo da classe de previsão: índice situação no curso na base_4
classe = base_4.iloc[:,14].values

#Transformação de atributos categóricos em numéricos
labelencoder_previsores2 = LabelEncoder() 
labelencoder_previsores4 = LabelEncoder()
labelencoder_previsores6 = LabelEncoder()
labelencoder_previsores9 = LabelEncoder()
labelencoder_previsores13 = LabelEncoder()
labelencoder_previsores14 = LabelEncoder()

previsores[:,2] = labelencoder_previsores2.fit_transform(previsores[:,2])
previsores[:,4] = labelencoder_previsores4.fit_transform(previsores[:,4])
previsores[:,6] = labelencoder_previsores6.fit_transform(previsores[:,6])
previsores[:,9] = labelencoder_previsores9.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores13.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder_previsores14.fit_transform(previsores[:,14])

#exportar label encoder para aplicar no modelo
with open("encoders-rna/!labelencoder-previsores2", "wb") as le2: 
    pickle.dump(labelencoder_previsores2, le2)

with open("encoders-rna/!labelencoder-previsores4", "wb") as le4: 
    pickle.dump(labelencoder_previsores4, le4)
    
with open("encoders-rna/!labelencoder-previsores6", "wb") as le6: 
    pickle.dump(labelencoder_previsores6, le6)

with open("encoders-rna/!labelencoder-previsores9", "wb") as le9: 
    pickle.dump(labelencoder_previsores9, le9)

with open("encoders-rna/!labelencoder-previsores13", "wb") as le13: 
    pickle.dump(labelencoder_previsores13, le13)

with open("encoders-rna/!labelencoder-previsores14", "wb") as le14: 
    pickle.dump(labelencoder_previsores14, le14)

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Dummy codificação de valores, onde cada variável se torna uma coluna
onehotencoder = ColumnTransformer(transformers=[
    ('OneHot', OneHotEncoder(), [2, 4, 6, 9, 13, 14])], remainder='passthrough')
#problema ao usar .toarray()
previsores = onehotencoder.fit_transform(previsores).astype(float)

with open("encoders-rna/!onehotencoder-previsores", "wb") as oe: 
    pickle.dump(onehotencoder, oe)

# #Escalonamento dos dados, para dispo-los em mesma base, evitando viés
min_max_scaler = MinMaxScaler()
previsores = min_max_scaler.fit_transform(previsores)

with open("encoders-rna/!min-max-scaler-previsores", "wb") as s: 
    pickle.dump(min_max_scaler, s)


from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

"""### ETAPA_5 Construção do modelo de RNA e treinamento"""

# Divisão da base em treinamento e testes 80% treinamento, 20% testes
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores, classe, test_size = 0.20, random_state=70)

# Classificação e aprendizado com keras
classificador = Sequential()
# cálculo de neurônios (38 atributos c/ onehotencoder + 1 saída) / 2 = 19.5 = 20 neurônios
#primeira camada 
classificador.add(Dense(units = 20, activation = 'relu',
                kernel_initializer = 'random_uniform', input_dim = 38))
#zerar 15% dos valores, para ajudar reduzir overfitting e alta variabilidade
classificador.add(Dropout(0.15))
#segunda camada, oculta
classificador.add(Dense(units = 20, kernel_initializer = 'random_uniform', activation= 'relu'))
#camada de saída 1 = sigmoid, 2+ softmax
classificador.add(Dropout(0.15))
classificador.add(Dense(units = 1, activation='sigmoid'))

#compilação da rede
classificador.compile(optimizer= 'sgd', loss= 'binary_crossentropy', metrics = ['binary_accuracy'])
# classificador.compile(optimizer= 'sgd', loss= 'binary_crossentropy', metrics = ['binary_accuracy'])
# otimizando manualmente
# otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
# classificador.compile(optimizer= otimizador, loss= 'binary_crossentropy', metrics = ['binary_accuracy'])

#configurações do treinamento, atualização de pesos a cada 10 registros 
# history = classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 150)
# history = classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 1, epochs = 25)
# history = classificador.fit(previsores, classe, batch_size = 10, epochs = 200, validation_split=0.20)
# history = classificador.fit(previsores, classe, batch_size = 10, epochs = 200, validation_split=0.20)
history = classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 200)


"""### ETAPA_6 Mensurar desempenho do modelo de RNA"""
# listar atributos em historyg
print(history.history.keys())
sns.set_style('darkgrid')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xticks(size = 12)
plt.xlabel('Época', size = 16)
plt.yticks(size = 12)
plt.ylabel('Perda', size = 16)
plt.legend(['Treinamento', 'Validação'], loc='upper right', fontsize=12)
plt.show()


plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.xticks(size = 12)
plt.xlabel('Época', size = 16)
plt.yticks(size = 12)
plt.ylabel('Acurácia', size = 16)
plt.legend(['Treinamento', 'Validação'], loc='upper left', fontsize=12)
plt.show()

# def criarRede():
#     #Classificação e aprendizado com keras
#     classificador = Sequential()
#         # cálculo de neurônios (38 atributos c/ onehotencoder + 1 saída) / 2 = 19.5 = 20 neurônios
#         #primeira camada 
#     classificador.add(Dense(units = 40, activation = 'relu',
#                             kernel_initializer = 'random_uniform', input_dim = 38))
#         #zerar 20% dos valores, para ajudar reduzir overfitting e alta variabilidade
#     classificador.add(Dropout(0.2))
#         #segunda camada, oculta
#     classificador.add(Dense(units = 40, kernel_initializer = 'random_uniform', activation= 'relu'))
#         #terceira camada, oculta
#     classificador.add(Dropout(0.2))
#     classificador.add(Dense(units = 20, kernel_initializer = 'random_uniform', activation= 'relu'))
#         #camada de saída 1 = sigmoid, 2+ softmax
#     classificador.add(Dropout(0.2))
#     classificador.add(Dense(units = 1, activation='sigmoid'))

#     #compilação da rede
#     # classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics = ['binary_accuracy'])
#     # otimizando manualmente
#     otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
#     classificador.compile(optimizer= otimizador, loss= 'binary_crossentropy',
#         metrics = ['binary_accuracy'])

#     return classificador

# classificador = KerasClassifier(build_fn = criarRede, epochs = 25, batch_size = 5)
# resultados = cross_val_score(estimator = classificador, 
#                              X = previsores, y = classe,
#                              cv = 10, scoring= 'accuracy')
# media = resultados.mean()
# desvio = resultados.std()
# #configurações do treinamento, atualização de pesos a cada 10 registros, ajuste de pesos 100 vezes
# # history = classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 150)
# history = classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 1, epochs = 25)
# # history = classificador.fit(previsores, classe, batch_size = 1, epochs = 15, validation_split=0.20)


# #relu, bath = 10, epochs 200, random, binary, neurons = 20, sgd , 12 horas 0.9767
# parametros = {'batch_size': [5, 10, 20],
#              'epochs': [25, 50, 75, 100, 200],
#              'optimizer': ['adam', 'sgd'],
#              'loos': ['binary_crossentropy', 'hinge'],
#              'kernel_initializer': ['random_uniform', 'normal'],
#              'activation': ['relu', 'tahn'],
#              'neurons': [20, 40, 80]}

# #segundo realizado relu, bath = 1, epochs 100, random, binary, neurons = 10, sgd , 6 horas 0.97537
# parametros = {'batch_size': [1, 10],
#              'epochs': [25, 50, 100, 200],
#              'optimizer': ['adam', 'sgd'],
#              'loos': ['binary_crossentropy'],
#              'kernel_initializer': ['random_uniform'],
#              'activation': ['relu'],
#              'neurons': [10, 20]}

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    #Classificação e aprendizado com keras
    classificador = Sequential()
    #primeira camada 
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer, input_dim = 38))
    #zerar 15% dos valores, para ajudar reduzir overfitting e alta variabilidade
    classificador.add(Dropout(0.15))
    #segunda camada, oculta
    classificador.add(Dense(
        units = neurons, kernel_initializer = kernel_initializer, activation = activation))   
    classificador.add(Dropout(0.15))
    #camada de saída 1 = sigmoid, 2+ softmax
    classificador.add(Dense(units = 1, activation='sigmoid'))

    #compilação da rede
    classificador.compile(optimizer = optimizer, loss = loos,
        metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size': [1, 5, 10, 20],
              'epochs': [25, 50, 75, 100, 200],
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tahn'],
              'neurons': [10, 20, 40, 80]}

grid_search = GridSearchCV(estimator=classificador, 
                           param_grid=parametros,
                           scoring='accuracy',
                           cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_

print(melhores_parametros)
print(melhor_resultado)

#mensuração por matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score

#criar variável de previsões
previsoes = classificador.predict(previsores_teste)

#define threshold (limiar) do resultado em 20%
previsoes_threshold = (previsoes > 0.2)

#matriz de confusão e cálculo de acurácia
acuracia = accuracy_score(classe_teste, previsoes_threshold)
matriz = confusion_matrix(classe_teste, previsoes_threshold)

#keras
keras_resultado = classificador.evaluate(previsores_teste, classe_teste)


"""###ETAPA_6: Salvar modelo treinado """


from sklearn.neural_network import MLPClassifier
import pickle

classificador_rede_neural = MLPClassifier(activation='relu', batch_size = 10, solver='sgd')
classificador_rede_neural.fit(previsores, classe)

# pickle.dump(classificador_rede_neural, open('!rede_neural_20211019.sav', 'wb'))


import pickle
#importar modelo
# rede_neural = pickle.load(open('!rede_neural_20211013.sav', 'rb'))

novo_registro = previsores[1]
novo_registro = novo_registro.reshape(1, -1)
novo_registro.shape

#1 = evasão 
resposta = rede_neural.predict(novo_registro)


if resposta[0] == 0:
    print('Não evasão')
else:
    print('Evasão')

probabilidade_resposta = rede_neural.predict_proba(novo_registro)
probabilidade_resposta

confianca = probabilidade_resposta.max()
confianca

base_4.values.min()
# base_4.columns




"""# tuning de parâmetros gridsearch"""

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


parametros = {'activation': ['relu', 'logistic', 'tahn'],
              'solver': ['adam', 'sgd'],
              'batch_size': [5, 10]}

grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parametros)
grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_

#1,10,56 logistic, batch = 10 e adam 0.9760
#5, 10, 15, 20, 56 = mas com batch = 5
print(melhores_parametros)
print(melhor_resultado)


"""# validação cruzada"""

from sklearn.model_selection import cross_val_score, KFold
resultados_rede_neural = []

for i in range(30):
    print(i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
        
    rede_neural = MLPClassifier(activation='logistic' , batch_size='10' , solver='adam')
    scores = cross_val_score(rede_neural, previsores, classe, cv = kfold)
    resultados_rede_neural.append(scores.mean())
