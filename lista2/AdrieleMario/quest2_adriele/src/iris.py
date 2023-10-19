from mlp import MLP
from plotagem import *
import pandas as pd


# fetch dataset
iris=pd.read_csv('iris.data')

# data (as pandas dataframes)
iris=pd.DataFrame(iris)
y=iris.iloc[:,-1]
y=np.array(y)
x=iris.iloc[:, :-1]
x=np.array(x)
iris = iris.values
neuronios = 1 #quantidade de neuronios
entradas = 4 #quantidade de entradas
saidas = 3 #quantidade de saidas
epochs = 150
eta = 0.01
d_possiveis = ['Iris-setosa','Iris-versicolor','Iris-virginica']

#VARREDURA DOS NEURONIOS
for i in range(11)
    print(f'Configuração com {neuronios} neuronios =============================================')
    contador = 1

    #Cria as 10MLP's
    iris_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]

    for iris in iris_list:
        print(f'T: {contador} -----------------------')
        iris.mlp()
        plot_MSE(iris,f'Iris - {neuronios}N1HL',contador)
        plot_ACC(iris,f'Iris - {neuronios}N1HL',contador)
        contador += 1

    plot_MSEt(iris_list,f'Iris - {neuronios}N1HL')
    plot_MSEv(iris_list,f'Iris - {neuronios}N1HL')
    plot_ACCt(iris_list,f'Iris - {neuronios}N1HL')
    plot_ACCv(iris_list,f'Iris - {neuronios}N1HL')
    plotMatrizConf(iris_list,'Iris - {neuronios}N1HL')
    plot_boxplot(iris_list,'Iris - {neuronios}N1HL')