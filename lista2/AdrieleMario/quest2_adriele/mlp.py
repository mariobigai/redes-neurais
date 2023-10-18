from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.optimizers import SGD


class MLP:
    def __init__(self,x,y,d,eta,neuronios,n_entrada,n_saida,epochs):
        self.x = x
        self.y = y
        self.d = d
        self.eta = eta
        self.neuronios = neuronios
        self.n_entrada = n_entrada
        self.n_saida = n_saida
        self.epochs = epochs
        self.mse_history = []
        self.val_mse_history = []

    def mlp(self):

        d1 = self.d[0]
        d2 = self.d[1]
        d3 = self.d[2]
        mapeamento = {d1: 0, d2: 1, d3: 2}
        self.y = np.array([mapeamento[label] for label in self.y])

        dados_normalizados = -1 + 2 * ((self.x - self.x.min(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0)))
        x_min = self.x.min().min()
        x_max = self.x.max().max()
        dif = x_max - x_min

        # Separar o conjunto de treinamento, validação e teste em 70/15/15
        x_train, x_temp, y_train, y_temp = train_test_split(dados_normalizados, self.y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        # Criar o modelo
        model = Sequential()
        model.add(Dense(self.neuronios, input_dim=self.n_entrada, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(self.n_saida, activation='softmax'))

        model.compile(optimizer=SGD(learning_rate=self.eta), loss=['sparse_categorical_crossentropy', 'mse'], metrics=['accuracy'])

        # Treinar o modelo
        self.historico = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=50, epochs=self.epochs, verbose=0)

        self.mse_history = self.historico.history['loss']  # valores do MSE durante o treinamento
        self.val_mse_history = self.historico.history['val_loss']  # valores do MSE durante o treinamento
        self.acc_history = self.historico.history['accuracy']  # valores do acuracia durante o treinamento

        self.min_epoch = np.argmin(self.val_mse_history)  # Encontra o índice da época com o menor valor de erro
        print(f"A época com o menor valor de erro na validação foi {self.min_epoch + 1} com valor {np.min(self.val_mse_history)}")

        print("MSE no teste: %.3f" % self.mse_history[-1])
        print("Acurácia no teste: %.3f" % self.acc_history[-1])

        # Avaliar o modelo no conjunto de teste
        self.resultado = model.evaluate(x_test, y_test, verbose = 0)

        print("MSE na validação: %.3f" % (self.resultado[0]))
        print("Acurácia na validação: %.3f" % (self.resultado[1]))

        # Fazer previsões no conjunto de teste
        previsoes = model.predict(x_test,verbose=0)

        previsoes = (((previsoes + 1) / 2) * dif) + x_min

        # Converter as previsões em classes (arredondamento para o valor mais próximo)
        classes_previstas = np.argmax(previsoes, axis=1)

        # Calcular a matriz de confusão
        self.matriz_confusao = confusion_matrix(y_test, classes_previstas)

        # Obter as classes únicas presentes nos rótulos verdadeiros e previstos
        self.classes = unique_labels(y_test, classes_previstas)