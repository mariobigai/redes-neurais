from mlp import MLP
from plotagem import *


# fetch dataset
wine=pd.read_csv('wine.data')

# data (as pandas dataframes)
wine=pd.DataFrame(wine)
y=wine.iloc[:,0]
y=np.array(y)
x=wine.iloc[:, 1:]
x=np.array(x)
wine = wine.values
neuronios = 15 #quantidade de neuronios
entradas = 13 #quantidade de entradas
saidas = 3 #quantidade de saidas
epochs = 150
eta = 0.01
d_possiveis = [1,2,3]

#Cria as 10MLP's
wine_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]
#Treina as 10 MLP's
[wine.mlp() for wine in wine_list]
plot_MSEt(wine_list,'Wine')
plot_MSEv(wine_list,'Wine')
plot_ACCt(wine_list,'Wine')
plot_ACCv(wine_list,'Wine')
plotMatrizConf(wine_list,'Wine')