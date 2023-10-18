from mlp import MLP
from plotagem import *
import pandas as pd


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

#FAZ A VARREDURA DOS NEURONIOS
for i in range(3):
    print(f'Configuração com {neuronios} neuronios =============================================')
    contador = 1
    #Cria as 10MLP's
    wine_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]
    #Treina as 10 MLP's
    for wine in wine_list:
        print(f'T: {contador} -----------------------')
        filepath = str(neuronios) + 'N1HL' + str(contador)
        wine.mlp(filepath)
        plot_MSE(wine, f'Wine - {neuronios}N1HL', contador)
        plot_ACC(wine, f'Wine - {neuronios}N1HL', contador)
        contador += 1


    plot_MSEt(wine_list,f'Wine - {neuronios}N1HL')
    plot_MSEv(wine_list,f'Wine - {neuronios}N1HL')
    plot_ACCt(wine_list,f'Wine - {neuronios}N1HL')
    plot_ACCv(wine_list,f'Wine - {neuronios}N1HL')
    plotMatrizConf(wine_list,f'Wine - {neuronios}N1HL')
    plot_boxplot(wine_list,f'Wine - {neuronios}N1HL')
    neuronios += 10
