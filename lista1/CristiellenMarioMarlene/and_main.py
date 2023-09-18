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

print(rede1.epocas, rede2.epocas)

x1 = np.linspace(-2,2)
x2_rede1 = (-rede1.pesos[0]-rede1.pesos[1]*x1)/(rede1.pesos[2])
x2_rede2 = (-rede2.pesos[0]-rede2.pesos[1]*x1)/(rede2.pesos[2])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter([0,0,1], [0,1,0], s=50, facecolor='C0', edgecolor='k', label='Saída 1')
ax.scatter([1], [1], s=50, facecolor='r', edgecolor='k', label='Saída 0')
ax.plot(x1, x2_rede1)
ax.plot(x1, x2_rede2)
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.1, 1.2)
ax.legend()
ax.grid()
plt.show()