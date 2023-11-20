import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class ELM:
    def __init__(self, neur_oculta, entrada, saida):

        self.neur_oculto = neur_oculta
        self.entrada = entrada
        self.saida = saida

        self.pesos_saida = None #--> pesos camada oculta -- saida
        self.bias_saida = None #--> bias camada saida

    def trat_dados(self):
        encoder = OneHotEncoder(sparse_output = False)
        encoderPlot = LabelEncoder()

        #Transforma as strings (Iris-Setosa, Iris-Versicolor, Iris-Virginica) -> (1, 2, 3)
        #Porém é preciso deixa em outro formato, como se fosse neuronio ligado e desligado
        # 1 0 0 --> Classe 1
        # 0 1 0 --> Classe 2
        # 0 0 1 --> Classe 3
        dadosSaida = encoder.fit_transform(np.array(self.saida).reshape(-1, 1))

        #Normalizacao da Entrada
        #Normalizados entre -1 à 1
        scaler = MinMaxScaler(feature_range = (-1, 1))

        entradaNormalizada = scaler.fit_transform(self.entrada)

        #Separa os dados em treino e teste
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(entradaNormalizada, dadosSaida, test_size = 0.2)

        #inicializa parametros
        self.neur_entrada = len(self.x_train[0])
        self.neur_saida = len(self.y_test[0])

        self.pesos_oculta = np.random.uniform(-1, 1, (self.neur_entrada, self.neur_oculto))
        self.bias_oculta = np.random.uniform(size = [self.neur_oculto])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        self.trat_dados()

        # Calcula a saída da camada oculta

        u = self.x_train.dot(self.pesos_oculta) + self.bias_oculta
        saida_oculta = self._sigmoid(u)

        #Calcula a pseudo-inversa de Moore-Penrose da saída da camada oculta
        pseudo_inversa = np.linalg.pinv(saida_oculta)

        self.pesos_saida = np.dot(pseudo_inversa, self.y_train)
        self.bias_saida = np.zeros(self.neur_saida)

        # Calcula a saída da camada oculta
        cam_escondida = self._sigmoid(self.x_train.dot(self.pesos_oculta) + self.bias_oculta)

        # Calcula a saída da rede
        saida_rede = np.dot(cam_escondida, self.pesos_saida) + self.bias_saida

        saidaPrev = np.argmax(saida_rede, axis=-1)
        saidaPret = np.argmax(self.y_train, axis=-1)

        mse = mean_squared_error(saidaPret, saidaPrev)
        acc = np.sum(saidaPrev == saidaPret) / len(self.y_train)

        return mse, acc

    def predict(self):
        # Calcula a saída da camada oculta
        cam_escondida = self._sigmoid(self.x_test.dot(self.pesos_oculta) + self.bias_oculta)

        # Calcula a saída da rede
        saida_rede = np.dot(cam_escondida, self.pesos_saida) + self.bias_saida

        return saida_rede

    def evaluate(self):
        saidaPrevista = self.predict()

        saidaPrev = np.argmax(saidaPrevista, axis = -1)
        saidaPret = np.argmax(self.y_test, axis = -1)

        mse = mean_squared_error(saidaPret, saidaPrev)
        acc = np.sum(saidaPrev == saidaPret) / len(self.y_test)

        return mse, acc




