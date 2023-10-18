from ucimlrepo import fetch_ucirepo
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.regularizers import l2
import seaborn as sns
from sklearn.metrics import mean_squared_error

def plot_MSE(rede,dados):
    plt.plot(rede.historico.history['loss'])
    plt.plot(rede.historico.history['val_loss'])
    plt.title(f'MSE por época - {dados}')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend(['Treino', 'Validação'])
    plt.savefig(f'MSE por Épocas - {dados} - Run único.png', dpi=500, format='png',orientation='portrait')
    plt.show()

def plot_MSEt(lista_de_rede, dados):
    cont_redes = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_rede:
        ax.plot(rede.historico.history['loss'], linestyle='-', linewidth=3, label=f'T: {cont_redes} - Perda: {rede.resultado[0]}')
        cont_redes += 1
    ax.legend()
    ax.grid()
    plt.title(f'MSE por época no treinamento - {dados}')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.savefig(f'MSE por Épocas T - {dados}.png', dpi=500, format='png',orientation='portrait')
    plt.show()

def plot_MSEv(lista_de_rede, dados):
    cont_redes = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_rede:
        ax.plot(rede.historico.history['val_loss'], linestyle='-', linewidth=3, label=f'T: {cont_redes}')
        cont_redes += 1
    ax.legend()
    ax.grid()
    plt.title(f'MSE por época na validacao - {dados}')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.savefig(f'MSE por Épocas V - {dados}.png', dpi=500, format='png', orientation='portrait')
    plt.show()

def plot_ACCt(lista_de_rede, dados):
    cont_redes = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_rede:
        ax.plot(rede.historico.history['accuracy'], linestyle='-', linewidth=3, label=f'T: {cont_redes}')
        cont_redes += 1
    ax.legend()
    ax.grid()
    plt.title(f'Acurácia por época no treinamento - {dados}')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.savefig(f'ACC por Épocas T - {dados}.png', dpi=500, format='png',orientation='portrait')
    plt.show()

def plot_ACCv(lista_de_rede, dados):
    cont_redes = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_rede:
        ax.plot(rede.historico.history['val_accuracy'], linestyle='-', linewidth=3, label=f'T: {cont_redes}')
        cont_redes += 1
    ax.legend()
    ax.grid()
    plt.title(f'Acurácia por época na validação - {dados}')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.savefig(f'ACC por Épocas V - {dados}.png', dpi=500, format='png',orientation='portrait')
    plt.show()

def plotMatrizConf(lista_de_rede, dados):
    contador = 1
    for rede in lista_de_rede:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Criar uma imagem da matriz de confusão
        cax = ax.matshow(rede.matriz_confusao, cmap=plt.cm.Blues)


        # Adicionar valores dentro das células da matriz de confusão
        for i in range(len(rede.classes)):
            for j in range(len(rede.classes)):
                plt.text(j, i, rede.matriz_confusao[i, j], ha='center', va='center', color='black')

        # Adicionar uma barra de cores
        plt.colorbar(cax)

        # Rotular os eixos com as classes reais e previstas
        ax.set_xticks(np.arange(len(rede.classes)))
        ax.set_yticks(np.arange(len(rede.classes)))
        ax.set_xticklabels(rede.classes)
        ax.set_yticklabels(rede.classes)

        # Adicionar rótulos aos eixos
        plt.xlabel('Previsto')
        plt.ylabel('Real')

        #Adicionar uma legenda
        plt.title(f'Matriz de Confusão - {dados} - T:{contador}')
        plt.savefig(f'Matriz de confusão  - {dados} - T: {contador}', dpi=500, format='png', orientation='portrait')
        plt.show()
        contador += 1