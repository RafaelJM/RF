import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2  # Usado para manipulação de imagem
import time
import csv

def plot(losses, total_rewards, elapsed_times, episode, game):
    # Após o loop de treinamento, você pode plotar os gráficos
    plt.figure(figsize=(15, 5))
    
    # Gráfico de perda
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Episódio')
    plt.ylabel('Loss')
    plt.title('Loss por Episódio')
    plt.legend()
    
    # Gráfico de recompensa total
    plt.subplot(1, 3, 2)
    plt.plot(total_rewards, label='Recompensa Total', color='orange')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.title('Recompensa Total por Episódio')
    plt.legend()
    
    # Gráfico de tempo decorrido
    plt.subplot(1, 3, 3)
    plt.plot(elapsed_times, label='Tempo Decorrido', color='green')
    plt.xlabel('Episódio')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo Decorrido por Episódio')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./{game}/stats.png', dpi=300)
    plt.show()
    
def save(model, losses, total_rewards, elapsed_times, episode, game):
    # Salvar os dados em um arquivo CSV
    with open(f'./{game}/training_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episódio', 'Loss', 'Recompensa Total', 'Tempo (segundos)'])  # Cabeçalho
        for episode in range(len(losses)-1):
            writer.writerow([episode + 1, losses[episode], total_rewards[episode], elapsed_times[episode]])
    
    model.save(f'./{game}/model_ep_{episode}_reward_{int(total_reward)}.keras')
    
def preprocess_state(state, shape):
    # Converter para tons de cinza e redimensionar para (84, 84)
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, shape)
    return np.expand_dims(resized_state, axis=-1)  # Adicionar a dimensão de canais

def compute_returns(need_calculation, historic, gamma=0.99):    
    G = 0
    for i in reversed(need_calculation):
        G = historic['rewards'][i] + gamma * G  # Acumula a recompensa futura
        historic['future_returns'][i] = G  # Atualiza diretamente o histórico

def train_discrete(model, optimizer, historic):
    # Obtem as iterações
    historic = historic[
        ~np.isnan(historic['rewards']) & ~np.isnan(historic['future_returns'])
    ]
    # Treina
    with tf.GradientTape() as tape:
        logits = model(historic["observations"])
        policy_distributions = tfp.distributions.Categorical(logits=logits)
        log_probs = policy_distributions.log_prob(historic["actions"])
    
        mean = tf.reduce_mean(historic["future_returns"])
        std = tf.maximum(tf.math.reduce_std(historic["future_returns"]), 1e-12)
        normalized_future_returns = (historic["future_returns"] - mean) / std
    
        loss = -tf.reduce_mean(log_probs * normalized_future_returns)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    return loss


def run_enviroment(env, model, save_folder, num_episodes = 1000, gamma=0.95, hitoric_size = 10000):
    state, _ = env.reset()
    env.render()
    
    dtype = np.dtype([
        ('observations', np.float32, (8)),  # Agora usando imagens de 84x84 pixels e 1 canal (tons de cinza)
        ('actions', np.int32),                
        ('rewards', np.float32),                
        ('future_returns', np.float32)          
    ])
    
    losses = []
    total_rewards = []
    elapsed_times = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    hcount = 0
    historic = np.zeros(hitoric_size, dtype=dtype)
    historic['rewards'] = np.nan
    historic['future_returns'] = np.nan

# Inicializando o ambiente CartPole-v1
game = 'LunarLander-v2'
env = gym.make(game, render_mode='human')

# Criando o modelo
model = create_model(env)

# Parâmetros de treinamento
num_episodes = 1000

dtype = np.dtype([
    ('observations', np.float32, (8)),  # Agora usando imagens de 84x84 pixels e 1 canal (tons de cinza)
    ('actions', np.int32),                
    ('rewards', np.float32),                
    ('future_returns', np.float32)          
])

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0      
    need_calculation = []  
    while True:
        # Prever a ação usando o modelo
        state_input = np.expand_dims(state, axis=0)
        action_probs = model.predict(state_input,verbose = 0)[0]
        action = np.random.choice(len(action_probs), p=action_probs)  # Escolha da ação com base na distribuição de probabilidade
        
        # Executando a ação no ambiente
        next_state, reward, done, info, _ = env.step(action)  # Corrigido aqui
        total_reward += reward
        
        historic[hcount] = (
            state,
            action,
            reward,
            np.nan
        )
        need_calculation.append(hcount)
        hcount = 0 if hcount == hitoric_size-1 else hcount+1
        
        # Atualizando o estado atual
        state = next_state
        
        # Se o episódio terminar, exibir resultados
        if done:
            start_time = time.time()
            compute_returns(need_calculation, historic)
            loss = train(model, optimizer, historic)
            elapsed_time = time.time() - start_time
            losses.append(loss)
            total_rewards.append(total_reward)
            elapsed_times.append(elapsed_time)
            print("Episódio {}/{} - Recompensa: {} - loss {:.4f} - hcount {} - train_time {:.4f}".format(episode + 1, num_episodes, total_reward, loss, hcount, elapsed_time))
            break
    if not episode%10:
        save(model, losses, total_rewards, elapsed_times, episode, game)
        plot(losses, total_rewards, elapsed_times, episode, game)
