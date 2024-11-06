import numpy as np
import random
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

class DeepSARSAAgent:
    def __init__(self, state_size, action_size, discount=0.95, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.discount = discount  # Discount factor for future rewards
        self.learning_rate = learning_rate  # Learning rate for Q-network updates
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.model = self.build_model()  # Initialize the Q-network

    def build_model(self):
        # Build a simple neural network for Q-value approximation
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self, batch_size):
        # Train the Q-network using experience replay
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, next_action, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount * self.model.predict(next_state)[0][next_action]
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
