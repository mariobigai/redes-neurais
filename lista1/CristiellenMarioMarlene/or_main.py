from perceptron import Perceptron
from adaline import Adaline
from plotagem import *

# Porta Lógica AND ----------------------------------------------------------------
entradas = [[0,0],
            [0,1],
            [1,0],
            [1,1]]
saidas = [-1,1,1,1]

bits_plot = [[0], [0], [0, 1, 1], [1, 0, 1]]
# ---------------------------------------------------------------------------------

#Cria 10 redes Perceptron e 10 redes Adaline
perceptron_list = [Perceptron(entradas, saidas, 100) for _ in range(10)]
adaline_list = [Adaline(entradas, saidas, 100) for _ in range(10)]

# Treina os 10 perceptrons e 10 adalines
[perceptron.treinar() for perceptron in perceptron_list]
[adaline.treinar() for adaline in adaline_list]

salva_10_runs(perceptron_list, 'PERCEPTRON', 'PORTA OR', bits_plot)
salva_10_runs(adaline_list, 'ADALINE', 'PORTA OR', bits_plot)

salva_melhor_de_run(perceptron_list, 'PERCEPTRON', 'PORTA OR', bits_plot)
salva_melhor_de_run(adaline_list, 'ADALINE', 'PORTA OR', bits_plot)

plota_MSE_grafico(perceptron_list, 'PERCEPTRON', 'PORTA OR', bits_plot)
plota_MSE_grafico(adaline_list, 'ADALINE', 'PORTA OR', bits_plot)