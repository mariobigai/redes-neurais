import random

class Perceptron:
    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, bias=1, w0=random.random()):
        self.amostras = amostras
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = 0
        self.bias = bias
        self.w0 = w0
        self.n_amostras = len(amostras)  # número de linhas (amostras)
        self.n_atributos = len(amostras[0])  # número de colunas (atributos)
        self.pesos = []

    ## Função sinal
    def sinal(self, u):
        if u <= 0:
            return 0
        return 1

    def treinar(self):
        # Adiciona o Bias nas amostras
        for amostra in self.amostras:
            amostra.insert(0, self.bias)
        # Gerar valores randômicos entre 0 e 1 (pesos) conforme o número de atributos
        for i in range(self.n_atributos):
            self.pesos.append(random.random())
        # Adiciona w0 no vetor de pesos
        self.pesos.insert(0, self.w0)

        while (self.epocas < 100):
            erro = False

            for i in range(self.n_amostras):
                # Inicializar potencial de ativação
                u = 0
                # Para cada atributo...
                for j in range(self.n_atributos + 1):
                    # Multiplicar amostra e seu peso e também somar com o potencial que já tinha
                    u += self.pesos[j] * self.amostras[i][j]
                # Obter a saída da rede considerando g a função sinal
                y = self.sinal(u)
                # Verificar se a saída da rede é diferente da saída desejada
                if y != self.saidas[i]:
                    # Calcular o erro
                    erro_aux = self.saidas[i] - y
                    # Fazer o ajuste dos pesos para cada elemento da amostra
                    for j in range(self.n_atributos + 1):
                        self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
                    # Atualizar variável erro, já que erro é diferente de zero (existe)
                    erro = True
            # Atualizar contador de épocas
            self.epocas += 1

            # Condição de parada erro inexistente ou 1000 épocas
            if not erro:
                break

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
