from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# fetch dataset
wine = pd.read_csv('wine.data')
wine = pd.DataFrame(wine)
y = wine.iloc[:, 0]
y = np.array(y)
x = wine.iloc[:, 1:]
x = np.array(x)
wine = wine.values

mapeamento = {1: 0, 2: 1, 3: 2}
y = [mapeamento[label] for label in y]
y = np.array(y)

dados_normalizados = -1 + 2 * ((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
x_min = x.min().min()
x_max = x.max().max()
dif = x_max - x_min

# Separar o conjunto de treinamento, validação e teste em 70/15/15
x_train, x_temp, y_train, y_temp = train_test_split(dados_normalizados, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Criar o modelo
model = Sequential()
model.add(Dense(15, input_dim=13, kernel_initializer='normal', activation='tanh'))
model.add(Dense(3, activation='softmax'))  # 3 unidades na camada de saída para classificação multiclasse

model.compile(optimizer='sgd', loss=['sparse_categorical_crossentropy', 'mse'], metrics=['accuracy'])

# Treinar o modelo
historico = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=50, epochs=150)

# Avaliar o modelo no conjunto de teste
resultado = model.evaluate(x_test, y_test)
#print("Perda do teste: %.3f" % (resultado[0]))

dados = 'Wine'

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend([f'Treino - Perda:{resultado[0]}', 'Validação'])
plt.savefig(f'MSE por Épocas T - {dados}', dpi=500, orientation='portrait')
plt.show()

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(['Treino', 'Validação'])
plt.savefig(f'ACC por Épocas', dpi=500, orientation='portrait')
plt.show()

mse_history = []
# Fazer previsões no conjunto de teste
previsoes = model.predict(x_test)

previsoes = (((previsoes + 1) / 2) * dif) + x_min

# Converter as previsões em classes (arredondamento para o valor mais próximo)
classes_previstas = np.argmax(previsoes, axis=1)

# Calcular a matriz de confusão
matriz_confusao = confusion_matrix(y_test, classes_previstas)

# Obter as classes únicas presentes nos rótulos verdadeiros e previstos
classes = unique_labels(y_test, classes_previstas)

# Criar uma figura e um eixo para o gráfico
plt.figure(figsize=(8, 8))
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
plt.savefig(f'Matriz de confusão', dpi=500, orientation='portrait')
plt.show()