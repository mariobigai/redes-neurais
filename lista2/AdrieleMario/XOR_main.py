import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

def generate_data_plot_points():
    # Gera uma grade de pontos para plotagem
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    points = np.c_[xx.ravel(), yy.ravel()]

    return points, xx, yy

def plot_decision_boundary(X, y, mlp, i, j, caminho):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    input_data = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.feedforward(input_data)
    Z = (Z > 0.5).astype(int)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=('red', 'blue'), alpha=0.8)

    # Destacar pontos de treinamento
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', marker='o', s=80, linewidth=1)

    plt.title(f"PORTA XOR: Fronteira de Separação - Run: {j} Neurônios HL: {i}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(caminho)
    plt.close('all')
    # plt.show()


def main():
    # Dados de entrada e saída - PORTA XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    for i in range(1, 6):
        if i == 1:
           arquitetura = '1N1HL'
        elif i == 2:
           arquitetura = '2N1HL'
        elif i == 3:
           arquitetura = '3N1HL'
        elif i == 4:
           arquitetura = '4N1HL'
        elif i == 5:
           arquitetura = '5N1HL'
        print(arquitetura + 50 * '-')
        for j in range(1, 11):
            mlp = MLP(layer_sizes=[2, i, 1], learning_rate=0.05)

            # Treinamento da MLP
            epochs = 10000
            mlp.train(X, y, epochs, target_mse=0.005)

            # Plotar gráfico MSE x épocas
            plt.plot(range(len(mlp.mse_history)), mlp.mse_history)
            plt.title(f'PORTA XOR: MSE x Épocas - Run: {j} Neurônios HL: {i}')
            plt.xlabel('Épocas')
            plt.ylabel('MSE')
            plt.savefig(arquitetura + f' - MSE_run{j}')
            plt.close('all')
            # plt.show()

            # Gerar pontos para a plotagem
            points, xx, yy = generate_data_plot_points()

            # Plotar a fronteira de separação
            plot_decision_boundary(X, y.flatten(), mlp, i, j, caminho=arquitetura + f' - FdS_run{j}')
            print(f'Run: {j} - Épocas: {len(mlp.mse_history)} -  MSE final: {mlp.mse_history[-1]}')

if __name__ == "__main__":
    main()