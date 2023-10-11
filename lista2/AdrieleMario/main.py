import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP  # Substitua 'mlp_module' pelo nome real do módulo que contém a classe MLP

def generate_data_plot_points():
    # Gera uma grade de pontos para plotagem
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    points = np.c_[xx.ravel(), yy.ravel()]

    return points, xx, yy

def plot_decision_boundary(X, y, mlp):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    input_data = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.feedforward(input_data)
    Z = np.round(Z)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Destacar pontos de treinamento
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', marker='o', s=80, linewidth=1)

    plt.title("Fronteira de Separação")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def main():
    # Dados de entrada e saída
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(layer_sizes=[2, 3, 1], learning_rate=0.1)

    # Treinamento da MLP
    epochs = 10000
    mlp.train(X, y, epochs, target_mse=0.01)

    # Plotar gráfico MSE x épocas
    plt.plot(range(len(mlp.mse_history)), mlp.mse_history)
    plt.title('MSE x Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.show()

    # Gerar pontos para a plotagem
    points, xx, yy = generate_data_plot_points()

    # Plotar a fronteira de separação
    plot_decision_boundary(X, y.flatten(), mlp)

if __name__ == "__main__":
    main()