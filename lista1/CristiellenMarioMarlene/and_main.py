import numpy as np
from matplotlib import pyplot as plt
from perceptron import Perceptron

entradas = [[0,0],
            [0,1],
            [1,0],
            [1,1]]
saidas = [0,0,0,1]

#Cria 10 redes Perceptron
perceptron_list = [Perceptron(entradas, saidas, 100) for _ in range(10)]

# Treina os 10 perceptrons
for perceptron in perceptron_list:
    perceptron.treinar()

x1 = np.linspace(-0.2,1.2)
cont_redes = 1

#Plotagem----------------------------------------------------------------------------
for perceptron in perceptron_list:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter([0,0,1], [0,1,0], s=100, facecolor='C0', edgecolor='k', label='C1 => bit0')
    ax.scatter([1], [1], s=100, facecolor='r', edgecolor='k', label='C2 => bit1')

    for i in range(perceptron.epocas - 1):
        # print(i)
        # print(rede1.pesos_hist[i][0], rede1.pesos_hist[i][1], rede1.pesos_hist[i][2])
        x2_rede1 = (-perceptron.pesos_hist[i][0] - perceptron.pesos_hist[i][1] * x1) / (perceptron.pesos_hist[i][2])
        ax.plot(x1, x2_rede1, linestyle = '--', linewidth = 1.5, color = 'black')

    x2_rede1 = (-perceptron.pesos_hist[-1][0] - perceptron.pesos_hist[-1][1] * x1) / (perceptron.pesos_hist[-1][2])
    ax.plot(x1, x2_rede1, linestyle = '-', linewidth = 3, color = 'green', label = f'Perceptron - Run: {cont_redes} - Época: {perceptron.epocas}')

    # ax.set_xlim(-0.1, 1.2)
    # ax.set_ylim(-0.2, 1.2)
    ax.legend()
    ax.grid()
    plt.show()
    cont_redes += 1