import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import cv2  # Usado para manipulação de imagem

# Inicializações
epsilon = 0.1
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_increment = 0.05
epsilon_decrement = 0.01
threshold = 0.6  # Ajuste conforme necessário

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
        self.epsilon = 1e-5  # Para evitar prioridades zero
        self.dissatisfaction = 0.01 #epsilon
        
    def adjust_exploration(self):
        future_returns = self.buffer['future_returns'][:self.size]
        priorities = self.buffer['priorities'][:self.size]

        # Encontrar a prioridade mínima e o retorno futuro correspondente
        imin_future_returns = np.argmin(future_returns)
        imax_future_returns = np.argmax(future_returns)
        
        self.dissatisfaction = max(0, min(
                self.dissatisfaction + 0.05 if priorities[imax_future_returns] > priorities[imin_future_returns] else self.dissatisfaction - 0.05
            , 0.8))
            
    def add(self, observation, action, reward):
        # Inicializar future_returns como NaN e prioridade como zero
        self.buffer[self.index] = (observation, action, reward, np.nan, 0.0)
        self.need_calculation.append(self.index)
        self.size = max(self.index + 1, self.size)
        self.index = (self.index + 1) % self.max_size

    def compute_returns(self, gamma):
        G = 0
        for i in reversed(self.need_calculation):
            G = self.buffer['rewards'][i] + gamma * G
            self.buffer['future_returns'][i] = G
        self.need_calculation = []
    
        # Obter o menor e maior retorno futuro no buffer
        valid_returns = self.buffer['future_returns'][:self.size]
        min_return = np.min(valid_returns)
        max_return = np.max(valid_returns)
        midpoint = (max_return + min_return) / 2  # Ponto médio entre o menor e o maior retorno
    
        # Configurar o valor de "a" para controlar a concavidade da função de segundo grau
        a = 1.0 / ((max_return - midpoint) ** 2)
    
        # Atualizar prioridades com base na função de segundo grau
        for i in range(self.size):
            G = self.buffer['future_returns'][i]
            
            # Função de segundo grau para criar o efeito "U" com raiz no ponto médio
            deviation_from_midpoint = a * (G - midpoint) ** 2  # Quanto mais longe do ponto médio, maior a prioridade
            
            self.buffer['priorities'][i] = deviation_from_midpoint + self.epsilon  # Adicionar epsilon para evitar zero



    def sample(self, batch_size):
        # Usar prioridades para calcular probabilidades de amostragem
        priorities = self.buffer['priorities'][:self.size]
        total_priority = np.sum(priorities)
        if total_priority == 0:
            # Se todas as prioridades forem zero (improvável), amostrar uniformemente
            probabilities = np.ones(self.size) / self.size
        else:
            probabilities = priorities / total_priority

        sampled_indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        samples = self.buffer[sampled_indices]
        return samples, sampled_indices, probabilities[sampled_indices]

def train(model, optimizer, replay_buffer, batch_size, beta=0.4):
    samples, indices, sampling_probs = replay_buffer.sample(batch_size)

    # Calcular pesos de importância
    weights = (1.0 / (replay_buffer.size * sampling_probs)) ** beta
    weights /= weights.max()  # Normalizar pesos para [0, 1]
    
    returns = samples['future_returns']
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8  # Evitar divisão por zero
    normalized_returns = (returns - mean_return) / std_return

    with tf.GradientTape() as tape:
        logits = model(samples['observations'])
        policy_distributions = tfp.distributions.Categorical(logits=logits)
        log_probs = policy_distributions.log_prob(samples['actions'])

        loss = -tf.reduce_mean(weights * log_probs * normalized_returns)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def preprocess_state(state, shape = (84, 84)):
    # Converter para tons de cinza e redimensionar para (84, 84)
    gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized_state = cv2.resize(gray_state, shape)
    return np.expand_dims(resized_state, axis=-1)  # Adicionar a dimensão de canais