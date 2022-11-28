import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

from DQN import DQN

# namedtuple and memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(object):
    def __init__(self, environment):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DQN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.target = DQN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        self.decay = 0.93
        self.randomness = 1.0
        self.min_randomness = 0.001

        self.n_actions = environment.action_space.n

    def act(self, state):

        if random.random() > self.randomness:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self, transitions):
        batch = Transition(*zip(*transitions))

        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        next_state_values = self.target(next_state_batch).max(1)[0].detach()
        expected_state_action_values = next_state_values + reward_batch

        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)

    def test_mode(self):
        self.randomness = 0.0
