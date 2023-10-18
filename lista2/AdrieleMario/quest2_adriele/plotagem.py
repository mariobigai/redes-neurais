import matplotlib.pyplot as plt
import numpy as np

def save_plot_as_image(fig, file_name):
    fig.savefig(file_name, dpi=500, format='png', orientation='portrait')

def plot_MSE(rede,dados,contador):
    plt.plot(rede.historico.history['loss'])
    plt.plot(rede.historico.history['val_loss'])
    plt.title(f'MSE por época - {dados} - Run:{contador}')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend(['Treino', f'Validação - Perda: {rede.resultado[0]}'])
    save_plot_as_image(plt, f'MSE_por_Epocas_{dados}_Run_{contador}.png')
    plt.show()

def plot_ACC(rede,dados,contador):
    plt.plot(rede.historico.history['accuracy'])
    plt.plot(rede.historico.history['val_accuracy'])
    plt.title(f'Acurácia por época - {dados} - Run:{contador}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend(['Treino', f'Validação - Acerto: {rede.resultado[1]}'])
    save_plot_as_image(plt, f'Acuracia_por_Epocas_{dados}_Run_{contador}.png')
    plt.show()

def plot_MSEt(lista_de_rede, dados):
    cont_redes = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_rede:
        ax.plot(rede.historico.history['loss'], linestyle='-', linewidth=3, label=f'T: {cont_redes} - Epoch: {rede.min_epoch+1} - {rede.mse_history[rede.min_epoch]}')
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
        ax.plot(rede.historico.history['val_loss'], linestyle='-', linewidth=3, label=f'T: {cont_redes} - Perda: {rede.resultado[0]}')
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
        ax.plot(rede.historico.history['val_accuracy'], linestyle='-', linewidth=3, label=f'T: {cont_redes} - Acerto: {rede.resultado[1]}')
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
        save_plot_as_image(plt, f'Matriz_de_Confusão{dados}_Run_{contador}.png')
        plt.show()
        contador += 1

def plot_boxplot(lista_de_rede, dados):
    perdas = [rede.resultado[0] for rede in lista_de_rede]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(perdas)
    ax.set_title(f'Boxplot do MSE - {dados}')
    ax.set_xticklabels(['MSE'])

    plt.savefig(f'Boxplot do MSE - {dados}.png', dpi=500, format='png', orientation='portrait')
    plt.show()