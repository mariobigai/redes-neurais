'''
Nome: letras.dat
Quantidade de memórias fundamentais: 12
Dimensão da grade: 20 x 20
'''
from hopfield import *
import matplotlib.pyplot as plt

# Números = [1, 2, 3, 4] representados por matriz(9,5) de pixeis (Branco:1, Preto:-1)
letras = pd.read_table('letras_teste.txt', sep=' ')

#Cria objeto hopfield
hp = hopfield(patterns=letras.values, noise_percentage=0.50,
              pattern_n_row=20, pattern_n_column=20, ib=0, epochs=100000, neta = 0.1)
hp.run()

perc = []
## Calculando Acertos e Erro médio
for elemento, out in zip(letras.values, hp.outputs.values):
    acerto = 0
    erro = 0
    print(100*'-')
    for i in range(hp.nrow*hp.ncol):
        if elemento[i] != out[i]:
            erro += 1
        else:
            acerto += 1
    perc.append(erro)

    print('Pixels certos: ' + str(acerto))
    print('Pixels errados: ' + str(erro))

total_perc = np.array(perc) * 100 / (hp.nrow*hp.ncol)
med_perc = np.mean(total_perc)
print(100*'-')
print(100*'-')
print(f'Erro medio %: {med_perc:.4f}')

# #Plotando resultados
fig, axs = plt.subplots(nrows=3, ncols=len(letras.values), figsize=(10, 15))
fig.suptitle(f'Hopfield (Livro) - {round((1-hp.noise)*100)}% de ruído')

for j in range(len(letras.values)):
    axs[0][j].set_title(f'Amostra {j+1}')
    axs[0][j].imshow(letras.values[j].reshape(hp.nrow,hp.ncol), cmap='Grays')

    axs[1][j].set_title(f'Ruido em {j+1}')
    axs[1][j].imshow(hp.noised_img.iloc[j,:].values.reshape(hp.nrow,hp.ncol), cmap='Grays')

    axs[2][j].set_title(f'Saída {j+1}')
    axs[2][j].imshow(hp.outputs.iloc[j,:].values.reshape(hp.nrow,hp.ncol), cmap='Grays')
plt.show()
