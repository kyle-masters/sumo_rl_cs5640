import torch.nn as nn

# DQN Model
class DQN(nn.Module):
    def __init__(self, obs_space, act_space):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_space),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
