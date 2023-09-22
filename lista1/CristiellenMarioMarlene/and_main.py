import numpy as np
from matplotlib import pyplot as plt
from perceptron import Perceptron

# Porta Lógica AND ----------------------------------------------------------------
entradas = [[0,0],
            [0,1],
            [1,0],
            [1,1]]
saidas = [0,0,0,1]
# ---------------------------------------------------------------------------------

#Cria 10 redes Perceptron
perceptron_list = [Perceptron(entradas, saidas, 100) for _ in range(10)]

# Treina os 10 perceptrons
[perceptron.treinar() for perceptron in perceptron_list]



# Plotagem----------------------------------------------------------------------------
x1 = np.linspace(-0.2,1.2)
## Primeira parte: Plota todos os separadores lineares das Runs
cont_redes = 0
for perceptron in perceptron_list:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.scatter([0,0,1], [0,1,0], s=100, facecolor='C0', edgecolor='k', label='$C_1$ = bit 0')
        ax.scatter([1], [1], s=100, facecolor='r', edgecolor='k', label='$C_2$ = bit 1')

        for i in range(perceptron.epocas - 1):
            # print(i)
            # print(rede1.pesos_hist[i][0], rede1.pesos_hist[i][1], rede1.pesos_hist[i][2])
            x2_rede1 = (-perceptron.pesos_hist[i][0] - perceptron.pesos_hist[i][1] * x1) / (perceptron.pesos_hist[i][2])
            if i == 0:
                ax.plot(x1, x2_rede1, linestyle = '--', linewidth = 1.5, color = 'black', alpha=0.5, label = 'Épocas Anteriores')
            else:
                ax.plot(x1, x2_rede1, linestyle='--', linewidth=1.5, color='black', alpha=0.5)

        x2_rede1 = (-perceptron.pesos_hist[-1][0] - perceptron.pesos_hist[-1][1] * x1) / (perceptron.pesos_hist[-1][2])
        ax.plot(x1, x2_rede1, linestyle = '-', linewidth = 3, color = 'green', label = f'Perceptron - Run: {cont_redes+1} - Época: {perceptron.epocas}')

        ax.set_xlim(-0.1, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.legend(loc='lower center')
        ax.grid()
        plt.title(f'PERCEPTRON - PORTA AND (Run: {cont_redes+1})')
        plt.xlabel('Entrada $x_1$')
        plt.ylabel('Entrada $x_2$')
        plt.savefig(f'Perceptron Run {cont_redes+1}', dpi=500, orientation='portrait')

        cont_redes += 1

## Segunda Parte: Plota o melhor separador de cada run
cont_redes = 0
fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter([0, 0, 1], [0, 1, 0], s=100, facecolor='C0', edgecolor='k', label='$C_1$ = bit 0')
ax.scatter([1], [1], s=100, facecolor='r', edgecolor='k', label='$C_2$ = bit 1')
for perceptron in perceptron_list:

    x2_rede1 = (-perceptron.pesos_hist[-1][0] - perceptron.pesos_hist[-1][1] * x1) / (perceptron.pesos_hist[-1][2])
    ax.plot(x1, x2_rede1, linestyle='-', linewidth=3, label=f'Run: {cont_redes+1}; Época: {perceptron.epocas}; MSE: {perceptron.MSE_valor:,.2f}')

    cont_redes += 1

ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.legend()
ax.grid()
plt.title('MELHORES PERCEPTRONS - PORTA AND')
plt.xlabel('Entrada $x_1$')
plt.ylabel('Entrada $x_2$')
plt.savefig(f'Best Perceptron', dpi=500, orientation='portrait')


## Terceira Parte: Plota curva do MSE
cont_redes = 0
fig, ax = plt.subplots(figsize=(15, 8))
for perceptron in perceptron_list:
    ax.plot(range(perceptron.epocas), perceptron.MSE_list, linestyle='-', linewidth=3, label=f'Run: {cont_redes}')
    cont_redes +=1
ax.legend()
ax.grid()
plt.title('Evolução do MSE')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.savefig(f'MSE Perceptron', dpi=500, orientation='portrait')