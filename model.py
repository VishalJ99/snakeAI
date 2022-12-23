import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        # init hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.policy = model
        self.target = copy.deepcopy(self.policy)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        Q_values = self.policy(state)
        Q_targets = Q_values.clone()

        for idx in range(len(done)):
            if done[idx]: 
                Q_target = reward[idx]
            else:
                Q_target = reward[idx] + self.gamma * torch.max(self.target(next_state[idx]).detach())

            Q_targets[idx][torch.argmax(action[idx]).item()] = Q_target
    
        self.optimizer.zero_grad()
        loss = self.criterion(Q_targets, Q_values)
        loss.backward()
        self.optimizer.step()

