from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import load_model

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

    def mlp(self, arq_run):
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

        self.y_test = y_test

        # Criar o modelo para o treinamento
        model = Sequential()
        model.add(Dense(self.neuronios, input_dim=self.n_entrada, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(self.n_saida, activation='softmax'))

        model.compile(optimizer=SGD(learning_rate=self.eta), loss=['sparse_categorical_crossentropy', 'mse'], metrics=['accuracy'])

        checkpoint_path = arq_run + '.hdf5'

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                         save_best_only=True,
                                         mode='min', verbose=0)
        # Treinar o modelo
        self.historico = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=self.epochs, verbose=0, callbacks=[cp_callback])

        self.mse_history = self.historico.history['loss']  # valores do MSE durante o treinamento
        self.val_mse_history = self.historico.history['val_loss']  # valores da MSE para validação durante o treinamento
        self.acc_history = self.historico.history['accuracy']  # valores do acuracia durante o treinamento
        self.val_acc_history = self.historico.history['val_accuracy'] # valores do acuracia para validação durante o treinamento

        self.min_epoch = np.argmin(self.val_mse_history)  # Encontra o índice da época com o menor valor de erro

        # Avaliar o modelo no conjunto de teste

        # carrega o ultimo checkpoint do treino
        model_test = load_model(checkpoint_path)

        self.resultado = model_test.evaluate(x_test, y_test, verbose = 0)

        # Fazer previsões no conjunto de teste
        previsoes = model_test.predict(x_test,verbose=0)

        # Desnormalização
        previsoes = (((previsoes + 1) / 2) * dif) + x_min

        # Converter as previsões em classes (arredondamento para o valor mais próximo)
        self.classes_previstas = np.argmax(previsoes, axis=1)

        # Calcular a matriz de confusão
        self.matriz_confusao = confusion_matrix(y_test, self.classes_previstas)

        # Obter as classes únicas presentes nos rótulos verdadeiros e previstos
        self.classes = unique_labels(y_test, self.classes_previstas)