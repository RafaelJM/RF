import numpy as np
import matplotlib.pyplot as plt
import csv

def moving_average(data, window_size):
    """Calcula a média móvel."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot(losses, total_rewards, elapsed_times, priorities_dif, episode, game):
    # Após o loop de treinamento, você pode plotar os gráficos
    plt.figure(figsize=(10, 10))  # Aumenta a altura para acomodar mais gráficos

    # Gráfico de perda
    plt.subplot(4, 1, 1)
    plt.plot(losses, label='Loss')
    if len(losses) >= 20:  # Certifica-se de que há dados suficientes
        plt.plot(np.arange(19, len(losses)), moving_average(losses, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.title('Loss por Episódio')
    plt.legend()

    # Gráfico de recompensa total
    plt.subplot(4, 1, 2)
    plt.plot(total_rewards, label='Recompensa Total', color='orange')
    if len(total_rewards) >= 20:
        plt.plot(np.arange(19, len(total_rewards)), moving_average(total_rewards, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensa Total por Episódio')
    plt.legend()

    # Gráfico de tempo decorrido
    plt.subplot(4, 1, 3)
    plt.plot(elapsed_times, label='Tempo Decorrido', color='green')
    if len(elapsed_times) >= 20:
        plt.plot(np.arange(19, len(elapsed_times)), moving_average(elapsed_times, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo Decorrido por Episódio')
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(priorities_dif, label='Diferença de Prioridades', color='purple')
    if len(priorities_dif) >= 20:
        plt.plot(np.arange(19, len(priorities_dif)), moving_average(priorities_dif, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Diferença de Prioridades')
    plt.title('Diferença de Prioridades por Episódio')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./output/{game}/stats.png', dpi=300)
    plt.show()

def save(model, losses, total_rewards, elapsed_times, episode, game):
    # Salvar os dados em um arquivo CSV
    with open(f'./output/{game}/training_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episódio', 'Loss', 'Recompensa Total', 'Tempo (segundos)'])  # Cabeçalho
        for episode in range(len(losses)-1):
            writer.writerow([episode + 1, losses[episode], total_rewards[episode], elapsed_times[episode]])