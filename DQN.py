import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define layers in the network
        self.fc1 = nn.Linear(input_size, 64)  # Adjust the output size of fc1 to match the input size
        self.fc2 = nn.Linear(64, 64)  # Adjust the input size of fc2 accordingly
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            print("cap reached")
            del self.memory[0]

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)


class DQNAgent:
    def __init__(self, input_size, output_size, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 min_epsilon=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # initialize target and policy networks
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(1000000)

        # evaluation
        self.loss_history = []
        self.stacks_history = []
        self.q_values_history = []
        self.feature_gradients = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output_size)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def optimize_model(self):
        if len(self.memory.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*transitions)  # Unpack transitions
        state_batch = torch.stack(state_batch)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)))
        non_final_next_states = torch.stack([s for s in next_state_batch if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        q_values = state_action_values.detach().numpy().reshape(-1)  # Flatten to store all Q-values in a single array
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()

        # evaluation
        input_gradients = self.policy_net.fc1.weight.grad.clone().detach().cpu().numpy()


        self.loss_history.append(loss.item())
        self.q_values_history.append(q_values)
        self.feature_gradients.append(input_gradients)


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)