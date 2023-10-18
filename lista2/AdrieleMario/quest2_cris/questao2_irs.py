import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Normalização dos dados entre -1 e 1
dados_normalizados = -1 + 2 * ((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
x_min = x.min()
x_max = x.max()
dif = x_max - x_min

mse_results = []
accuracy_results = []
previsao_test = []

# Executar o código 10 vezes
for rodada in range(1):
    # Separar o conjunto de treinamento, validação e teste (você pode ajustar as proporções)
    x_train, x_temp, y_train, y_temp = train_test_split(dados_normalizados, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Criar e treinar o modelo (você pode ajustar o modelo e os hiperparâmetros)
    model = Sequential()
    model.add(Dense(5, input_dim=4, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='sgd', loss=['sparse_categorical_crossentropy', 'mse'], metrics=['accuracy'])
    model_checkpoint_callback = ModelCheckpoint(filepath = './tmp/run.ckpt', monitor='val_loss', save_weigth_only=True,
                                                mode='min', save_best_only=True, verbose=1)

    historico = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=5, epochs=100, verbose=1, callbacks=[model_checkpoint_callback])

    # Avaliar o modelo
    resultado = model.evaluate(x_test, y_test)
    mse_results.append(historico.history['loss'])  # MSE é a métrica de índice 1
    accuracy_results.append(historico.history['accuracy'])  # Acurácia é a métrica de índice 2

    # Avaliar o modelo no conjunto de teste
    model.load_weights('./tmp/run.ckpt')
    resultado = model.evaluate(x_test, y_test)
    print("Perda do teste: %.3f" % (resultado[0]))

    previsoes = model.predict(x_test,
                              batch_size=None,
                              verbose='auto',
                              steps=None,
                              callbacks=None,
                              max_queue_size=10,
                              workers=1,
                              use_multiprocessing=False)

    # pegar o maior valor de cada linha e subtrair 1, para encontrar diferença
    maiores_valores = np.max(previsoes, axis=1)

    # Subtraia 1 de cada um dos maiores valores
    resultados_finais = 1 - maiores_valores

    previsao_test.append(resultados_finais)

    # Converter as previsões em classes
    classes_previstas = np.argmax(previsoes, axis=1)

    # Calcular a matriz de confusão
    matriz_confusao = confusion_matrix(y_test, classes_previstas)

    # Obter as classes únicas presentes nos rótulos verdadeiros e previstos
    classes = unique_labels(y_test, classes_previstas)

    # Criar uma figura e um eixo para o gráfico
    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    # Criar uma imagem da matriz de confusão
    cax = ax.matshow(matriz_confusao, cmap=plt.cm.Blues)

    # Adicionar uma barra de cores
    plt.colorbar(cax)

    # Rotular os eixos com as classes reais e previstas
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Adicionar rótulos aos eixos
    plt.xlabel('Previsto')
    plt.ylabel('Real')

    # Adicionar valores dentro das células da matriz de confusão
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, matriz_confusao[i, j], ha='center', va='center', color='black')

    # Adicionar uma legenda
    plt.title('Matriz de Confusão')
    plt.show()

    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    plt.xlabel('épocas')
    plt.ylabel('acurácia')
    plt.legend(['Treino', 'Validação'])
    plt.show()

    mse_treino = np.array(historico.history['loss'])
    mse_treino = (((mse_treino + 1) / 2) * dif) + x_min
    mse_treino = mse_treino.tolist()

    mse_val = np.array(historico.history['val_loss'])
    mse_val = (((mse_val + 1) / 2) * dif) + x_min
    mse_val = mse_val.tolist()

    plt.plot(mse_treino)
    plt.plot(mse_val)
    plt.xlabel('épocas')
    plt.ylabel('perda')
    plt.legend(['Treino', 'Validação'])
    plt.show()

mse_results = np.array(mse_results)
mse_results = (((mse_results + 1) / 2) * dif) + x_min
mse_results = mse_results.tolist()

# Crie um gráfico de boxplot dos resultados de MSE
plt.figure(figsize=(4, 6))
plt.boxplot(previsao_test)
plt.xlabel('Rodada')
plt.ylabel('MSE')
plt.title('Boxplot dos Resultados de MSE-teste')
plt.show()