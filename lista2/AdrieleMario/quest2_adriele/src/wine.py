from mlp import MLP
from plotagem import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# fetch dataset
wine=pd.read_csv('wine.data')

# data (as pandas dataframes)
wine=pd.DataFrame(wine)
y=wine.iloc[:,0]
y=np.array(y)
x=wine.iloc[:, 1:]
x=np.array(x)
wine = wine.values
neuronios = 10 #quantidade de neuronios
entradas = 13 #quantidade de entradas
saidas = 3 #quantidade de saidas
epochs = 150
eta = 0.01
d_possiveis = [1,2,3]
lista_de_arquiteturas = []
nome_das_arquiteturas = ['10N1HL', '20N1HL']

#FAZ A VARREDURA DOS NEURONIOS
for i in range(2):
    print(f'{neuronios}N1HL' + 60 * '--')
    contador = 1
    #Cria as 10MLP's
    wine_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]
    #Treina as 10 MLP's
    for wine in wine_list:
        filepath = str(neuronios) + 'N1HL-run_' + str(contador) + '_'
        wine.mlp(filepath)
        print('Previstas:' + str(wine.classes_previstas))
        print('Reais: '+ str(wine.y_test))
        print(f'Run{contador} - '+
              f'Melhor MSE da validação: {np.min(wine.val_mse_history):.3f} na época {wine.min_epoch + 1}; ' +
              f'MSE no treino: {wine.mse_history[-1]:.3f}; '+
              f'Acc no treino: {wine.val_acc_history[-1]:.3f}; '+
              f'MSE do teste: {wine.resultado[0]:.3f}; '+
              f'Acc do teste: {wine.resultado[1]:.3f}; ')
        plot_MSE(wine, f'Wine - {neuronios}N1HL', contador)
        plot_ACC(wine, f'Wine - {neuronios}N1HL', contador)
        contador += 1
    plot_MSEt(wine_list, f'Wine - {neuronios}N1HL')
    plot_MSEv(wine_list, f'Wine - {neuronios}N1HL')
    plot_ACCt(wine_list, f'Wine - {neuronios}N1HL')
    plot_ACCv(wine_list, f'Wine - {neuronios}N1HL')
    neuronios += 10
    lista_de_arquiteturas.append(wine_list)

plot_boxplot(lista_de_arquiteturas, nome_das_arquiteturas)