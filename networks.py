import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dims, n_actions, fc1_dims=256, fc2_dims=256, fc3_dims=256, name='ActorNet',
                 chkpt_dir='tmp/ddpg', max_action=1):
        super(ActorNetwork, self).__init__()

        self.max_action = max_action
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.mu = nn.Linear(fc3_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = T.tanh(self.mu(x)) * self.max_action

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dims, n_actions, fc1_dims=256, fc2_dims=256, fc3_dims=256, name="CriticNet",
                 chkpt_dir='tmp/ddpg', max_action=1):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.q = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta) 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))