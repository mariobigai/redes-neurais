import numpy as np
from matplotlib import pyplot as plt
from perceptron import Perceptron
from adaline import Adaline

entradas = [[0,0],
            [0,1],
            [1,0],
            [1,1]]
saidas = [0,0,0,1]

rede1 = Perceptron(entradas, saidas)
rede2 = Adaline(entradas, saidas)

rede1.treinar()
rede2.treinar()

x1 = np.linspace(0,1)
x2 = (-rede1.pesos[0]-rede1.pesos[1]*x1)/(rede1.pesos[2])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter([0,0,1], [0,1,0], s=50, facecolor='C0', edgecolor='k')
ax.scatter([1], [1], s=50, facecolor='r', edgecolor='k')
ax.plot(x1, x2)
plt.show()