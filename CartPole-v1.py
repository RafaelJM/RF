import gym
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers, models
import main
import utils

# Definindo a arquitetura do modelo
def model(env):
    model = models.Sequential()

    # Camadas densas para processamento de vetores de estado
    model.add(layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)))  # 24 neurônios na primeira camada
    model.add(layers.Dense(24, activation='relu'))  # Segunda camada densa
    model.add(layers.Dense(env.action_space.n, activation='softmax'))  # Número de ações depende do ambiente

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    return model


# Inicializando o ambiente CartPole-v1
game = 'CartPole-v1'
env = gym.make(game, render_mode='human')
state, _ = env.reset()
env.render()

# Criando o modelo
model = model(env)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
gamma=0.99
batch_size = 4096
epsilon = 0.1

# Inicialize as listas para armazenar os valores
losses = []
total_rewards = []
elapsed_times = []
epsilons = []
priorities_dif = []

replay_buffer = main.ReplayBuffer(1000000, (4))

max_reward = 200

# Parâmetros de treinamento
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0      
    while True:        
        #if np.random.rand() < replay_buffer.dissatisfaction:  # Exploração
        #    action = np.random.choice(env.action_space.n)
        #    replay_buffer.dissatisfaction -= 0.01
        #else:
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
        if done or total_reward > max_reward:
            start_time = time.time()
            replay_buffer.compute_returns(gamma)
            loss = main.train(model, optimizer, replay_buffer, batch_size)
            replay_buffer.adjust_exploration()
            elapsed_time = time.time() - start_time
            losses.append(loss)
            total_rewards.append(total_reward)
            elapsed_times.append(elapsed_time)
            
            future_returns = replay_buffer.buffer['future_returns'][:replay_buffer.size]
            priorities = replay_buffer.buffer['priorities'][:replay_buffer.size]
    
            # Encontrar a prioridade mínima e o retorno futuro correspondente
            imin_future_returns = np.argmin(future_returns)
            imax_future_returns = np.argmax(future_returns)
            
            priorities_dif.append(priorities[imax_future_returns] - priorities[imin_future_returns])
            if priorities[imax_future_returns] - priorities[imin_future_returns] < -0.5:
                max_reward += 200
            print("Episódio {}/{} - Recompensa: {} - loss {:.4f} - memorias {} - train_time {:.4f} - dissatisfaction {:.3f}".format(episode + 1, num_episodes, total_reward, loss, replay_buffer.index, elapsed_time, replay_buffer.dissatisfaction))
            break
    if not episode%1:
        utils.save(model, losses, total_rewards, elapsed_times, episode, game)
        utils.plot(losses, total_rewards, elapsed_times, replay_buffer, priorities_dif, episode, game)
        
