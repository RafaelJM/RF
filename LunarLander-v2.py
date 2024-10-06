import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2  # Usado para manipulação de imagem
import time
import csv
from tensorflow.keras import regularizers  # Importar regularizers

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.buffer = np.zeros(max_size, dtype=[
            ('observations', np.float32, input_shape),
            ('actions', np.int32),
            ('rewards', np.float32),
            ('future_returns', np.float32),
            ('priorities', np.float32)
        ])
        self.index = 0
        self.size = 0
        self.need_calculation = []

    def add(self, observation, action, reward):
        self.buffer[self.index] = (observation, action, reward, np.nan, 1.0) # Inicializa a prioridade como 1.0 (ou outra constante)
        self.need_calculation.append(self.index)
        self.size = max(self.index+1,self.size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        # Amostragem das experiências com base nas prioridades
        probabilities = self.buffer["priorities"][:self.size] / np.sum(self.buffer["priorities"][:self.size])
        sampled_indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        
        # Retorna as amostras e suas prioridades
        return self.buffer[sampled_indices], sampled_indices

    def update_priorities(self, indices, new_priorities):
        self.buffer["priorities"][indices] = new_priorities
        
    def compute_returns(self, gamma):        
        G = 0
        for i in reversed(self.need_calculation):
            G = self.buffer['rewards'][i] + gamma * G  # Acumula a recompensa futura
            self.buffer['future_returns'][i] = G  # Atualiza diretamente o histórico
        self.need_calculation = []

def train(model, optimizer, replay_buffer, batch_size, beta=0.4):
    historic, sampled_indices = replay_buffer.sample(batch_size)
    
    # Calcular os pesos das amostras
    probabilities = historic["priorities"] / np.sum(historic["priorities"])
    weights = (len(historic) * probabilities) ** (-beta)
    weights /= weights.max()  # Normaliza os pesos
    
    with tf.GradientTape() as tape:
        logits = model(historic["observations"])
        policy_distributions = tfp.distributions.Categorical(logits=logits)
        log_probs = policy_distributions.log_prob(historic["actions"])
    
        mean = tf.reduce_mean(historic["future_returns"] )
        std = tf.maximum(tf.math.reduce_std(historic["future_returns"]), 1e-12)
        normalized_future_returns = (historic["future_returns"] - mean) / std
    
        loss = -tf.reduce_mean(weights * log_probs * normalized_future_returns)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calcule novas prioridades baseadas nas perdas
        new_priorities = np.abs(loss.numpy()) + 1e-5  # Adicione um pequeno valor para evitar zero
        replay_buffer.update_priorities(sampled_indices, new_priorities)
    return loss

def preprocess_state(state):
    # Converter para tons de cinza e redimensionar para (84, 84)
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, (84, 84))
    return np.expand_dims(resized_state, axis=-1)  # Adicionar a dimensão de canais

# Definindo a arquitetura do modelo
def create_model(env):
    model = models.Sequential()

    # Camadas densas para processamento de vetores de estado
    model.add(layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)))  # 24 neurônios na primeira camada
    model.add(layers.Dense(24, activation='relu'))  # Segunda camada densa
    model.add(layers.Dense(env.action_space.n, activation='softmax'))  # Número de ações depende do ambiente

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    return model

def save(model, losses, total_rewards, elapsed_times, episode, game):
    # Salvar os dados em um arquivo CSV
    with open(f'./{game}/training_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episódio', 'Loss', 'Recompensa Total', 'Tempo (segundos)'])  # Cabeçalho
        for episode in range(len(losses)-1):
            writer.writerow([episode + 1, losses[episode], total_rewards[episode], elapsed_times[episode]])
    
    model.save(f'./{game}/model_ep_{episode}_reward_{int(total_reward)}.keras')

def moving_average(data, window_size):
    """Calcula a média móvel."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot(losses, total_rewards, elapsed_times, episode, game):
    # Após o loop de treinamento, você pode plotar os gráficos
    plt.figure(figsize=(10, 10))  # Aumenta a altura para acomodar mais gráficos

    # Gráfico de perda
    plt.subplot(3, 1, 1)
    plt.plot(losses, label='Loss')
    if len(losses) >= 20:  # Certifica-se de que há dados suficientes
        plt.plot(np.arange(19, len(losses)), moving_average(losses, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.title('Loss por Episódio')
    plt.legend()

    # Gráfico de recompensa total
    plt.subplot(3, 1, 2)
    plt.plot(total_rewards, label='Recompensa Total', color='orange')
    if len(total_rewards) >= 20:
        plt.plot(np.arange(19, len(total_rewards)), moving_average(total_rewards, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensa Total por Episódio')
    plt.legend()

    # Gráfico de tempo decorrido
    plt.subplot(3, 1, 3)
    plt.plot(elapsed_times, label='Tempo Decorrido', color='green')
    if len(elapsed_times) >= 20:
        plt.plot(np.arange(19, len(elapsed_times)), moving_average(elapsed_times, 20), color='black', linestyle='--', label='Média Móvel (20)')
    plt.xlabel('Episódio')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo Decorrido por Episódio')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./{game}/stats.png', dpi=300)
    plt.show()

# Inicializando o ambiente CartPole-v1
game = 'LunarLander-v2'
env = gym.make(game, render_mode='human')
state, _ = env.reset()
env.render()

# Criando o modelo
model = create_model(env)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
gamma=0.95
batch_size = 1024
epsilon = 0.1

# Inicialize as listas para armazenar os valores
losses = []
total_rewards = []
elapsed_times = []
epsilons = []

replay_buffer = ReplayBuffer(1000000, (8))

# Parâmetros de treinamento
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0      
    while True:
        if np.random.rand() < epsilon:  # Exploração
            action = np.random.choice(env.action_space.n)
        else:
            state_input = np.expand_dims(state, axis=0)
            action_probs = model.predict(state_input,verbose = 0)[0]
            action = np.random.choice(len(action_probs), p=action_probs)  # Escolha da ação com base na distribuição de probabilidade
            
        # Executando a ação no ambiente
        next_state, reward, done, info, _ = env.step(action)  # Corrigido aqui
        total_reward += reward
        
        replay_buffer.add(state, action, reward)
        
        # Atualizando o estado atual
        state = next_state
        
        # Se o episódio terminar, exibir resultados
        if done or total_reward < -500:
            start_time = time.time()
            replay_buffer.compute_returns(gamma)
            loss = train(model, optimizer, replay_buffer, batch_size)
            elapsed_time = time.time() - start_time
            losses.append(loss)
            total_rewards.append(total_reward)
            elapsed_times.append(elapsed_time)
            print("Episódio {}/{} - Recompensa: {} - loss {:.4f} - memorias {} - train_time {:.4f}".format(episode + 1, num_episodes, total_reward, loss, replay_buffer.index, elapsed_time))
            break
    if not episode%10:
        save(model, losses, total_rewards, elapsed_times, episode, game)
        plot(losses, total_rewards, elapsed_times, episode, game)
        
