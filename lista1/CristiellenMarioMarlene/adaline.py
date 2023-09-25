import random
import numpy as np

class Adaline:
    def __init__(self, amostras, saidas, max_epocas, taxa_aprendizado=0.1, bias=1, w0=random.random()):
        self.n_amostras = len(amostras)  # número de linhas (amostras)
        self.n_atributos = len(amostras[0])  # número de colunas (atributos)
        self.erros_list = [0] * self.n_amostras

        self.bias = bias
        self.w0 = w0
        self.pesos = []
        self.pesos_hist = []

        self.amostras = self.inicializa_amostras(amostras)
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = 0
        self.max_epocas = max_epocas

        self.erro = True

        self.sum_erro_list = []
        self.MSE_list = []

    ## Função sinal
    def sinal(self, u):
        if u <= 0:
            return -1
        return 1

    def adiciona_bias(self, lista_aux):
        for amostra in lista_aux:
            amostra.insert(0, self.bias)

    def inicializa_amostras(self, amostras):
        lista_aux = []
        entradas = amostras
        for amostra in entradas:
            lista = []
            for j in range(self.n_atributos):
                lista.append(amostra[j])
            lista_aux.append(lista)
        self.adiciona_bias(lista_aux)
        return lista_aux

    def inicializa_pesos(self):
        for i in range(self.n_atributos):
            self.pesos.append(random.random())
        # Adiciona w0 no vetor de pesos - Peso do bias
        self.pesos.insert(0, self.w0)
        self.pesos_hist.append(self.pesos)

    def guarda_peso_atual(self):
        peso_atual = [0] * (self.n_atributos + 1)
        for j in range(self.n_atributos + 1):
            peso_atual[j] = self.pesos[j]
        self.pesos_hist.append(peso_atual)

    def calcula_discriminante(self, i):
        # Inicializar discriminante
        u = 0
        # Para cada atributo (incluindo bias)
        for j in range(self.n_atributos + 1):
            # Multiplicar amostra e seu peso e também somar com o potencial que já tinha
            u += self.pesos[j] * self.amostras[i][j]
        return u

    def treinar(self):
        # Gerar valores randômicos entre 0 e 1 (pesos) conforme o número de atributos (incluindo peso do bias)
        self.inicializa_pesos()

        self.MSE_list.append(0)

        diff_MSE = 100

        # Condição de parada erro inexistente ou 100 épocas
        while (self.epocas < self.max_epocas and diff_MSE > 0.01):
            # Guarda os pesos ajustados referente a época:
            self.guarda_peso_atual()

            #Percorre as amostras
            for i in range(self.n_amostras):
                # calcula o discriminante
                u = self.calcula_discriminante(i)

                # calcula o erro relacionado a amostra
                erro_aux = self.saidas[i] - u
                self.erros_list[i] = erro_aux

                # Fazer o ajuste dos pesos para cada amostra (incluindo o peso do bias)
                for j in range(self.n_atributos + 1):
                    self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]

            # Faz somatória dos erros
            sum_erro = np.sum([abs(erro) for erro in self.erros_list])
            sum_erro_quad = np.sum([erro**2 for erro in self.erros_list])

            # Guarda somatória do erro relacionado a época
            self.sum_erro_list.append((sum_erro))

            # Guarda erro quadrático médio
            self.MSE_list.append((sum_erro_quad)/len(self.erros_list))

            # Verifica se a somatória do erro é aceitável
            diff_MSE = abs(self.MSE_list[-1] - self.MSE_list[self.epocas])

            # Atualizar contador de épocas
            self.epocas += 1

        # Cálculo do erro quadrático médio - levando em conta as iterações
        self.MSE_valor = (1/self.epocas) * np.sum(self.MSE_list)

    def teste(self, amostra):
        # Insere o bias na amostra
        amostra.insert(0, self.bias)
        # Inicializar potencial de ativação
        u = 0
        # Para cada atributo...
        for i in range(self.n_atributos + 1):
            # Multiplicar amostra e seu peso e também somar com o potencial que já tinha
            u += self.pesos[i] * amostra[i]
        # Obter a saída da rede considerando g a função sinal
        y = self.sinal(u)
        print('Classe: %d' % y)
