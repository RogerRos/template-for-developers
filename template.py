import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Uso del agente
state_size = 4  # Ejemplo
action_size = 2  # Ejemplo
agent = DQNAgent(state_size, action_size)

# Entrenamiento del agente (simplificado)
for e in range(1000):  # Número de episodios
    state = np.reshape(env.reset(), [1, state_size])  # Resetear entorno y obtener estado inicial
    for time in range(500):  # Limitar el número de pasos por episodio
        action = agent.act(state)  # Elegir acción
        next_state, reward, done, _ = env.step(action)  # Ejecutar acción en el entorno
        next_state = np.reshape(next_state, [1, state_size])  # Redimensionar siguiente estado
        agent.remember(state, action, reward, next_state, done)  # Guardar transición en la memoria
        state = next_state  # Actualizar estado actual
        if done:
            print(f"Episode: {e}/{1000}, score: {time}, epsilon: {agent.epsilon:.2}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)  # Reentrenar el modelo con experiencias pasadas

agent.save("dqn_model.h5")  # Guardar pesos del modelo
