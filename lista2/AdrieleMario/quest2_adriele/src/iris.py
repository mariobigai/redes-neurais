from mlp import MLP
from plotagem import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# fetch dataset
iris=pd.read_csv('iris.data')

# data (as pandas dataframes)
iris=pd.DataFrame(iris)
y=iris.iloc[:,-1]
y=np.array(y)
x=iris.iloc[:, :-1]
x=np.array(x)
iris = iris.values
neuronios = 10 #quantidade de neuronios
entradas = 4 #quantidade de entradas
saidas = 3 #quantidade de saidas
epochs = 1000
eta = 0.01
d_possiveis = ['Iris-setosa','Iris-versicolor','Iris-virginica']
lista_de_arquiteturas = []
nome_das_arquiteturas = ['10N1HL', '20N1HL', '30N1HL', '40N1HL', '50N1HL']

#VARREDURA DOS NEURONIOS
for i in range(len(nome_das_arquiteturas)):
    print(f'{neuronios}N1HL' + 60 * '--')
    contador = 1
    #Cria as 10MLP's
    iris_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]
    for iris in iris_list:
        filepath = str(neuronios) + 'N1HL-run_' + str(contador) + '_'
        iris.mlp(filepath)
        print('Previstas:' + str(iris.classes_previstas))
        print('Reais: ' + str(iris.y_test))
        print(f'Run{contador} - ' +
              f'Melhor MSE da validação: {np.min(iris.val_mse_history):.3f} na época {iris.min_epoch + 1}; ' +
              f'MSE no treino: {iris.mse_history[-1]:.3f}; ' +
              f'Acc no treino: {iris.val_acc_history[-1]:.3f}; ' +
              f'MSE do teste: {iris.resultado[0]:.3f}; ' +
              f'Acc do teste: {iris.resultado[1]:.3f}; ')
        plot_MSE(iris, f'Wine - {neuronios}N1HL', contador)
        plot_ACC(iris, f'Wine - {neuronios}N1HL', contador)
        contador += 1

    plot_MSEt(iris_list, f'Wine - {neuronios}N1HL')
    plot_MSEv(iris_list, f'Wine - {neuronios}N1HL')
    plot_ACCt(iris_list, f'Wine - {neuronios}N1HL')
    plot_ACCv(iris_list, f'Wine - {neuronios}N1HL')
    neuronios += 10
    lista_de_arquiteturas.append(iris_list)

plot_boxplot(lista_de_arquiteturas, nome_das_arquiteturas)