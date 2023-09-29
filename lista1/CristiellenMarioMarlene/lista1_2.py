import pandas as pd
from perceptron import Perceptron
from adaline import Adaline
from plotagem import *

#abrindo dataframe de dados de treinamento
dados_treino = pd.read_excel('lista1.xlsx')
dados_teste = pd.read_excel('Tabela3.3.xlsx')

y_train = dados_treino['d'].tolist()
y_train=[int(y) for y in y_train]

dados_treino = dados_treino.drop(columns=['d'])

x_train = dados_treino.values.tolist()
x_teste = dados_teste.values.tolist()

perceptron_list = [Perceptron(x_train, y_train, 5000, 0.001) for _ in range(5)]
adaline_list = [Adaline(x_train, y_train, 5000, 0.001) for _ in range(5)]

[perceptron.treinar() for perceptron in perceptron_list]
[adaline.treinar() for adaline in adaline_list]

y_teste_ada = []
y_teste_per = []

cont_percep = 0
cont_ada = 0

print(perceptron_list)
print(adaline_list)

for perceptron in perceptron_list:
    y_teste_per.append([perceptron.teste(amostra) for amostra in x_teste])
    print(f'Perceptron:{100 * "-"}')
    print(f'Épocas: {perceptron.epocas}')
    print(f'MSE final: {perceptron.MSE_list[perceptron.epocas - 1]}')
    print(f'Pesos Iniciais: {perceptron.pesos_hist[1]}')
    print(f'Pesos Final: {perceptron.pesos}')
    print(f'Teste: {y_teste_per[cont_percep]}')
    cont_percep += 1

for adaline in adaline_list:
    y_teste_ada.append([adaline.teste(amostra) for amostra in x_teste])
    print(f'Adaline:{100*"-"}')
    print(f'Épocas: {adaline.epocas}')
    print(f'MSE final: {adaline.MSE_list[adaline.epocas-1]}')
    print(f'Pesos Iniciais: {adaline.pesos_hist[1]}')
    print(f'Pesos Final: {adaline.pesos}')
    print(f'Teste: {y_teste_ada[cont_ada]}')
    cont_ada += 1

plota_MSE_grafico(perceptron_list, 'PERCEPTRON', 'MSE_quest2')
plota_MSE_grafico(adaline_list, 'ADALINE', 'MSE_quest2')


