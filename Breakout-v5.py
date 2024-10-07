import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time
from tensorflow.keras import regularizers  # Importar regularizers
import main
import utils

def create_model(env):
    model = models.Sequential()

    # Camadas convolucionais para processamento de imagens
    model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', 
                            input_shape=(84, 84, 1)))  # Sem L2
    model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))  # Sem L2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Sem L2
    model.add(layers.Flatten())

    # Camadas densas para decisões complexas
    model.add(layers.Dense(512, activation='relu'))  # Sem L2
    model.add(layers.Dense(env.action_space.n, activation='softmax'))  # Sem L2

    # Compilando o modelo com Adam e sparse categorical crossentropy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()

    return model

# Inicializando o ambiente CartPole-v1
game = 'Breakout-v5'
env = gym.make('ALE/'+game, render_mode='human')
state, _ = env.reset()
env.render()
started_lives = env.ale.lives()

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

replay_buffer = main.ReplayBuffer(500000, (84,84,1))

# Parâmetros de treinamento
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0      
    while True:
        processed_state = main.preprocess_state(state)
        if np.random.rand() < epsilon:  # Exploração
            action = np.random.choice(env.action_space.n)
        else:
            state_input = np.expand_dims(processed_state, axis=0)
            action_probs = model.predict(state_input,verbose = 0)[0]
            action = np.random.choice(len(action_probs), p=action_probs)  # Escolha da ação com base na distribuição de probabilidade
        
        # Executando a ação no ambiente
        next_state, reward, done, info, _ = env.step(action)  # Corrigido aqui
        total_reward += reward
        
        replay_buffer.add(processed_state, action, reward)
        
        # Atualizando o estado atual
        state = next_state
        
        # Se o episódio terminar, exibir resultados
        if done or started_lives != env.ale.lives():
            start_time = time.time()
            replay_buffer.compute_returns(gamma)
            loss = main.train(model, optimizer, replay_buffer, batch_size)
            elapsed_time = time.time() - start_time
            losses.append(loss)
            total_rewards.append(total_reward)
            elapsed_times.append(elapsed_time)
            print("Episódio {}/{} - Recompensa: {} - loss {:.4f} - memorias {} - train_time {:.4f}".format(episode + 1, num_episodes, total_reward, loss, replay_buffer.index, elapsed_time))
            break
    if not episode%10:
        utils.save(model, losses, total_rewards, elapsed_times, episode, game)
        utils.plot(losses, total_rewards, elapsed_times, replay_buffer, episode, game)
        