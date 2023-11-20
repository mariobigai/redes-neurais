'''
Nome: livro.dat
Quantidade de memórias fundamentais: 8
Dimensão da grade: 12 x 10
'''
from hopfield import *
import matplotlib.pyplot as plt

# Números = [1, 2, 3, 4] representados por matriz(9,5) de pixeis (Branco:1, Preto:-1)
livro = pd.read_table('livros.txt', sep=' ')

#Cria objeto hopfield
hp = hopfield(patterns=livro.values, noise_percentage=0.50,
              pattern_n_row=12, pattern_n_column=10, ib=0, epochs=100000, neta = 0.1)
hp.run()

## Calculando Acertos e Erro médio
for elemento, out in zip(livro.values, hp.outputs.values):
    acerto = 0
    erro = 0
    print(100*'-')
    for i in range(hp.nrow*hp.ncol):
        if elemento[i] != out[i]:
            erro += 1
        else:
            acerto += 1

    print('Pixels certos: ' + str(acerto))
    print('Pixels errados: ' + str(erro))
    med_perc = erro * 100 / (hp.nrow * hp.ncol)
    print(f'Erro medio %: {med_perc:.4f}')



#Plotando resultados
fig, axs = plt.subplots(nrows=3, ncols=8, figsize=(10, 15))
for j in range(len(livro.values)):
    axs[0][j].set_title(f'Amostra {j+1}')
    axs[0][j].imshow(livro.values[j].reshape(hp.nrow,hp.ncol), cmap='Grays')

    axs[1][j].set_title(f'Ruido em {j+1}')
    axs[1][j].imshow(hp.noised_img.iloc[j,:].values.reshape(hp.nrow,hp.ncol), cmap='Grays')

    axs[2][j].set_title(f'Saída {j+1}')
    axs[2][j].imshow(hp.outputs.iloc[j,:].values.reshape(hp.nrow,hp.ncol), cmap='Grays')
plt.show()

plt.show()