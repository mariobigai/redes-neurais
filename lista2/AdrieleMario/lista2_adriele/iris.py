from mlp import MLP
from plotagem import *


# fetch dataset
iris=pd.read_csv('iris.data')

# data (as pandas dataframes)
iris=pd.DataFrame(iris)
y=iris.iloc[:,-1]
y=np.array(y)
x=iris.iloc[:, :-1]
x=np.array(x)
iris = iris.values
neuronios = 4 #quantidade de neuronios
entradas = 4 #quantidade de entradas
saidas = 3 #quantidade de saidas
epochs = 150
eta = 0.01
d_possiveis = ['Iris-setosa','Iris-versicolor','Iris-virginica']

#Cria as 10MLP's
iris_list = [(MLP(x,y,d_possiveis,eta,neuronios,entradas,saidas,epochs)) for _ in range(10)]
#Treina as 10 MLP's
[iris.mlp() for iris in iris_list]
plot_MSEt(iris_list,'Iris')
plot_MSEv(iris_list,'Iris')
plot_ACCt(iris_list,'Iris')
plot_ACCv(iris_list,'Iris')
plotMatrizConf(iris_list,'Iris')