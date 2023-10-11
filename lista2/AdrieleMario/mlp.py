import numpy as np

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, bias_init_value=1.0):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        # Inicialização dos pesos e bias para todas as camadas
        # Aqui estamos considerando o BIAS não como uma entrada a mais, mas sim um termo independente
        self.weights = [np.random.uniform(-1,1,(layer_sizes[i-1], layer_sizes[i])) for i in range(1, self.num_layers)]
        self.biases = [np.random.uniform(-1,1,(1,layer_sizes[i]) ) for i in range(1, self.num_layers)]

        # Inicialização do histórico do MSE
        self.mse_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.activations = [X]
        self.outputs = [X]

        # Calcula a saída para cada camada
        for i in range(self.num_layers - 1):
            activation = np.dot(self.outputs[i], self.weights[i]) + self.biases[i]
            output = self.sigmoid(activation)

            self.activations.append(activation)
            self.outputs.append(output)

        return self.outputs[-1]

    def backpropagation(self, X, y, output):
        errors = [y - output]

        # Calcula os erros retropropagados
        for i in range(self.num_layers - 2, 0, -1):
            error = errors[-1].dot(self.weights[i].T)
            errors.append(error * self.sigmoid_derivative(self.outputs[i]))

        errors.reverse()

        # Atualiza os pesos e bias usando gradiente descendente
        for i in range(self.num_layers - 1):
            self.weights[i] += self.outputs[i].T.dot(errors[i]) * self.learning_rate
            self.biases[i] += np.sum(errors[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs, target_mse):
        for epoch in range(epochs):
            # Feedforward
            output = self.feedforward(X)

            # Backpropagation
            self.backpropagation(X, y, output)

            # Calcular e armazenar o erro quadrático médio (MSE)
            mse = np.mean(np.square(y - output))
            self.mse_history.append(mse)

            # Imprime o erro a cada 1000 épocas
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, MSE: {mse}')

            # Verifica se atingiu a taxa de precisão desejada
            if mse < target_mse:
                print(f'Atingiu a taxa de precisão desejada (MSE < {target_mse}). Parando o treinamento na época {epoch}.')
                break

# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada e saída
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Criação da MLP com 2 camadas ocultas (4 neurônios cada)
    mlp = MLP(layer_sizes=[2, 3, 1], learning_rate=0.01)

    # Treinamento da MLP com critério de parada baseado em MSE
    target_mse = 0.001
    mlp.train(X, y, epochs=100000, target_mse=target_mse)

    # Teste da MLP
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = mlp.feedforward(test_input)
    print("Predictions:")
    print(predictions)

    # # Imprimir o histórico de erro
    # print("MSE History:")
    # print(mlp.mse_history)