import numpy as np
from matplotlib import pyplot as plt

def plota_rede(rede, nome_da_rede, porta_logica, bits_plot):
    x1 = np.linspace(-0.2, 1.2)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(bits_plot[0], bits_plot[1], s=100, facecolor='C0', edgecolor='k', label='$C_1$ = bit 0')
    ax.scatter(bits_plot[2], bits_plot[3], s=100, facecolor='r', edgecolor='k', label='$C_2$ = bit 1')

    for i in range(rede.epocas - 1):
        x2_rede1 = (-rede.pesos_hist[i][0] - rede.pesos_hist[i][1] * x1) / (rede.pesos_hist[i][2])
        if i == 0:
            ax.plot(x1, x2_rede1, linestyle='--', linewidth=1.5, color='black', alpha=0.5, label='Épocas Anteriores')
        else:
            ax.plot(x1, x2_rede1, linestyle='--', linewidth=1.5, color='black', alpha=0.5)

    x2_rede1 = (-rede.pesos_hist[-1][0] - rede.pesos_hist[-1][1] * x1) / (rede.pesos_hist[-1][2])
    ax.plot(x1, x2_rede1, linestyle='-', linewidth=3, color='green',
            label=f'{nome_da_rede} - Época: {rede.epocas}')

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='lower center')
    ax.grid()
    plt.title(f'{nome_da_rede} - {porta_logica}')
    plt.xlabel('Entrada $x_1$')
    plt.ylabel('Entrada $x_2$')
    plt.show()


def salva_10_runs(lista_de_redes, nome_da_rede, porta_logica, bits_plot):
    x1 = np.linspace(-0.2,1.2)
    cont_redes = 0
    for rede in lista_de_redes:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.scatter(bits_plot[0], bits_plot[1], s=100, facecolor='C0', edgecolor='k', label='$C_1$ = bit 0')
            ax.scatter(bits_plot[2], bits_plot[3], s=100, facecolor='r', edgecolor='k', label='$C_2$ = bit 1')

            for i in range(rede.epocas - 1):
                x2_rede1 = (-rede.pesos_hist[i][0] - rede.pesos_hist[i][1] * x1) / (rede.pesos_hist[i][2])
                if i == 0:
                    ax.plot(x1, x2_rede1, linestyle = '--', linewidth = 1.5, color = 'black', alpha=0.5, label = 'Épocas Anteriores')
                else:
                    ax.plot(x1, x2_rede1, linestyle='--', linewidth=1.5, color='black', alpha=0.5)

            x2_rede1 = (-rede.pesos_hist[-1][0] - rede.pesos_hist[-1][1] * x1) / (rede.pesos_hist[-1][2])
            ax.plot(x1, x2_rede1, linestyle = '-', linewidth = 3, color = 'green', label = f'{nome_da_rede} - Run: {cont_redes+1} - Época: {rede.epocas}')

            ax.set_xlim(-0.1, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.legend(loc='lower center')
            ax.grid()
            plt.title(f'{nome_da_rede} - {porta_logica} (Run: {cont_redes+1})')
            plt.xlabel('Entrada $x_1$')
            plt.ylabel('Entrada $x_2$')
            plt.savefig(f'{nome_da_rede} Run {cont_redes+1}', dpi=500, orientation='portrait')
            plt.close(fig)
            cont_redes += 1

def salva_melhor_de_run(lista_de_redes, nome_da_rede, porta_logica, bits_plot):
    x1 = np.linspace(-0.2, 1.2)
    cont_redes = 0
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(bits_plot[0], bits_plot[1], s=100, facecolor='C0', edgecolor='k', label='$C_1$ = bit 0')
    ax.scatter(bits_plot[2], bits_plot[3], s=100, facecolor='r', edgecolor='k', label='$C_2$ = bit 1')
    for rede in lista_de_redes:

        x2_rede1 = (-rede.pesos_hist[-1][0] - rede.pesos_hist[-1][1] * x1) / (rede.pesos_hist[-1][2])
        ax.plot(x1, x2_rede1, linestyle='-', linewidth=3, label=f'Run: {cont_redes+1}; Época: {rede.epocas}; MSE: {rede.MSE_valor:,.2f}')

        cont_redes += 1

    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.legend()
    ax.grid()
    plt.title(f'MELHORES {nome_da_rede} - {porta_logica}')
    plt.xlabel('Entrada $x_1$')
    plt.ylabel('Entrada $x_2$')
    plt.savefig(f'Bests_{nome_da_rede}', dpi=500, orientation='portrait')
    plt.close(fig)

def plota_MSE_grafico(lista_de_redes, nome_da_rede, porta_logica, bits_plot):
    if nome_da_rede == 'ADALINE':
        for rede in lista_de_redes:
            rede.MSE_list.pop(0)


    cont_redes = 0
    fig, ax = plt.subplots(figsize=(15, 8))
    for rede in lista_de_redes:
        ax.plot(range(rede.epocas), rede.MSE_list, linestyle='-', linewidth=3, label=f'Run: {cont_redes}')
        cont_redes +=1
    ax.legend()
    ax.grid()
    plt.title('Evolução do MSE')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.savefig(f'{nome_da_rede} - {porta_logica}', dpi=500, orientation='portrait')