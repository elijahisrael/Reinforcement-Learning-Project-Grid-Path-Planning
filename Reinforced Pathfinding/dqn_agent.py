# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import json
from collections import deque


class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0005, gamma=0.99, epsilon=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

       
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.total_episodes = 0

        self.model = DuelingDQN(state_size, 128, action_size)
        self.target_model = DuelingDQN(state_size, 128, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                next_action = torch.argmax(self.model(next_state_tensor)).item()
                target_q = self.target_model(next_state_tensor)[0, next_action]
                target = reward if done else reward + self.gamma * target_q

            current_q = self.model(state_tensor)[0, action]
            target_tensor = torch.tensor([target], dtype=torch.float32)
            loss = self.criterion(current_q.unsqueeze(0), target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        self.total_episodes += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = min(max(self.epsilon, self.epsilon_min), 0.99)

    def save_model(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)
        metadata = {
            'epsilon': self.epsilon,
            'episodes': self.total_episodes
        }
        meta_path = path.replace(".pth", "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

    def load_model(self, path="dqn_model.pth"):
        try:
            self.model.load_state_dict(torch.load(path, weights_only=False))
            self.update_target_model()
            print(f"Model loaded successfully from {path}")
            meta_path = path.replace(".pth", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    self.epsilon = metadata.get('epsilon', self.epsilon_start)
                    self.total_episodes = metadata.get('episodes', 0)
                    print(f"Loaded metadata: Episodes={self.total_episodes}, Epsilon={self.epsilon}")
        except Exception as e:
            print(f"Failed to load model: {e}")
