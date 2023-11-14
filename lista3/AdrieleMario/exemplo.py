# Exemplo Números (slides)
from hopfield import *
import matplotlib.pyplot as plt

# Números = [1, 2, 3, 4] representados por matriz(9,5) de pixeis (Branco:1, Preto:-1)
exemplo = pd.read_table('exemplo.txt', sep=',')

#Cria objeto hopfield
hp = hopfield(patterns=exemplo.values, noise_percentage=0.08,
              pattern_n_row=9, pattern_n_column=5, ib=0, epochs=100000, neta = 0.1)
hp.run()

perc = []
## Calculando Acertos e Erro médio
for elemento, out in zip(exemplo.values, hp.outputs.values):
    acerto = 0
    erro = 0
    print(elemento)
    for i in range(9*5):
        if elemento[i] != out[i]:
            erro += 1
        else:
            acerto += 1
    perc.append(erro)

    print('Pixels certos: ' + str(acerto))
    print('Pixels errados: ' + str(erro))

total_perc = np.array(perc) * 100 / (9 * 5)
med_perc = np.mean(total_perc)
print(f'Erro medio %: {med_perc:.4f}')

#Plotando resultados
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 15))
fig.suptitle(f'Hopfield (Exemplo) - {round((1-hp.noise)*100)}% de ruído')
# ------- N1 -------

axs[0][0].set_title('Amostra 1')
axs[0][0].imshow(exemplo.values[0].reshape(hp.nrow,hp.ncol))

axs[1][0].set_title('Amostra 1 com Ruído')
axs[1][0].imshow(hp.noised_img.iloc[0,:].values.reshape(9,5))

axs[2][0].set_title('Saída 1')
axs[2][0].imshow(hp.outputs.iloc[0,:].values.reshape(9,5))


# ------- N2 -------
axs[0][1].set_title('Amostra 2')
axs[0][1].imshow(exemplo.values[1].reshape(9,5))

axs[1][1].set_title('Amostra 2 com Ruído')
axs[1][1].imshow(hp.noised_img.iloc[1,:].values.reshape(9,5))

axs[2][1].set_title('Saída 2')
axs[2][1].imshow(hp.outputs.iloc[1,:].values.reshape(9,5))


# ------- N3 -------
axs[0][2].set_title('Amostra 3')
axs[0][2].imshow(exemplo.values[2].reshape(9,5))

axs[1][2].set_title('Amostra 3 com Ruído')
axs[1][2].imshow(hp.noised_img.iloc[2,:].values.reshape(9,5))

axs[2][2].set_title('Saída 3')
axs[2][2].imshow(hp.outputs.iloc[2,:].values.reshape(9,5))


# ------- N4 -------
axs[0][3].set_title('Amostra 4')
axs[0][3].imshow(exemplo.values[3].reshape(9,5))

axs[1][3].set_title('Amostra 4 com Ruído')
axs[1][3].imshow(hp.noised_img.iloc[3,:].values.reshape(9,5))

axs[2][3].set_title('Saída 4')
axs[2][3].imshow(np.sign(hp.outputs.iloc[3,:].values.reshape(9,5)))

plt.show()