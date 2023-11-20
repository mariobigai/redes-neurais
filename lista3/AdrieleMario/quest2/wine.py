from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import pandas as pd
from ELMmain import ELM

# fetch dataset
wine = load_wine()

df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
dataset = wine.data

entrada = dataset[:, 0 : -1].astype('float')
saida = wine.target
neuronios = 5  # Número de neurônios na camada oculta

nomes_arq = []

f_treino = open('wine_results_treino.txt','a')
f_teste = open('wine_results_teste.txt','a')

mse_list_treino = [[] for _ in range(10)]
mse_list_teste = [[] for _ in range(10)]

#varredura dos neurônios
for i in range(10):
    # Criação e treinamento da ELM
    contador = 1
    f_treino.write(f'\nConfiguracao com {neuronios} neuronios =============================================')
    f_teste.write(f'\nConfiguracao com {neuronios} neuronios =============================================')

    #elm_iris = ELM(neuronios, entrada, saida)
    elm_iris_list =[(ELM(neuronios,entrada,saida))for _ in range(10)]
    nomes_arq.append(f'{neuronios}N1HL')

    for elm_iris in elm_iris_list:

        mse_treino, acc_treino = elm_iris.train()
        mse_teste, acc_teste = elm_iris.evaluate()

        mse_list_treino[i].append(mse_treino)
        mse_list_teste[i].append(mse_teste)

        f_treino.write(f'\nRun: {contador} - Acuracia: {acc_treino}')
        f_teste.write(f'\nRun: {contador} - Acuracia: {acc_teste}')

        if contador < 10:
            contador += 1

    neuronios += 10

fig_treino, ax_treino = plt.subplots(figsize=(8, 6))
ax_treino.boxplot(mse_list_treino)
ax_treino.set_xticklabels(nomes_arq)
ax_treino.set_title(f'Boxplot do MSE dos treino')
plt.savefig(f'Boxplot_wine_treino.png', dpi=500, format='png', orientation='portrait')
plt.show()
plt.close(fig_treino)

fig_teste, ax_teste = plt.subplots(figsize=(8, 6))
ax_teste.boxplot(mse_list_teste)
ax_teste.set_xticklabels(nomes_arq)
ax_teste.set_title(f'Boxplot do MSE dos testes')
plt.savefig(f'Boxplot_wine_teste.png', dpi=500, format='png', orientation='portrait')
plt.show()
plt.close(fig_teste)






