import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import deque, namedtuple
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.board_2048 import Board_2048

Data = namedtuple('Data', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = Data(state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data])).squeeze()
        action = torch.tensor(np.array([x[1] for x in data]))
        reward = torch.tensor(np.array([x[2] for x in data]))
        next_state = torch.tensor(np.stack([x[3] for x in data])).squeeze()
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))

        return state, action, reward, next_state, done



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        d = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, d, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, d, kernel_size=3, padding=0)


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = torch.cat((out1, out2), dim=1)

        return out


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.fc1 = nn.Linear(2048 * 16, 1024)
        self.fc2 = nn.Linear(1024, action_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training = self.training)
        out = self.fc2(out)

        return out



class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon_start = 0.9
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9999
        self.step = 0
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 4

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(self.device)
        self.qnet_target = QNet(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def encode_state(self, board):
        board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]
        board_tensor = torch.LongTensor(board_flat)
        board_one_hot = F.one_hot(board_tensor, num_classes=16).float().flatten()
        board_encoded = board_one_hot.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)

        return board_encoded

    def get_action(self, state):
        epsilon_threshold = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** self.step))
        self.step += 1
        if np.random.rand() < epsilon_threshold:

            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(self.encode_state(state))

            return qs.max(1)[1][0].to("cpu")

    def get_action_exploitation(self, state):
        qs = self.qnet(self.encode_state(state))

        return qs.max(1)[1][0].to("cpu")

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(self.encode_state(state), action, reward, self.encode_state(next_state), done)

        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss().to(self.device)
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet.train()



if __name__ == '__main__':
    env = Board_2048()
    agent = DQNAgent()
    agent.qnet.train()
    agent.qnet_target.eval()

    episodes = 500
    sync_interval = 20
    total_score_history, max_num_history = [], []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            is_action = env.is_action(action)
            if not is_action:
                continue
            else:
                next_state, reward, done = env.step(action)

                agent.update(state, action, reward, next_state, done)
                state = next_state

        if episode % sync_interval == 0:
            agent.sync_qnet()

        total_score, max_num = env.get_scores()

        total_score_history.append(total_score)
        max_num_history.append(max_num)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Score: {total_score}, Max Num: {max_num}")
            env.render_board()

            torch.save(agent.qnet.state_dict(), '/home/gjh020717/ai_ws/2048withDeepRL/model/qnet.pth')
            torch.save(agent.qnet_target.state_dict(), '/home/gjh020717/ai_ws/2048withDeepRL/model/qnet_target.pth')

    torch.save(agent.qnet.state_dict(), '/home/gjh020717/ai_ws/2048withDeepRL/model/qnet.pth')
    torch.save(agent.qnet_target.state_dict(), '/home/gjh020717/ai_ws/2048withDeepRL/model/qnet_target.pth')

    plt.subplot(211)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.plot(range(len(total_score_history)), total_score_history)

    plt.subplot(212)
    plt.xlabel('Episode')
    plt.ylabel('Max Number')
    plt.plot(range(len(max_num_history)), max_num_history)

    plt.show()

    with torch.no_grad():
        agent.qnet.eval()

        state = env.reset()
        done = False

        while not done:
            action = agent.get_action_exploitation(state)
            is_action = env.is_action(action)
            if not is_action:
                continue
            else:
                state, reward, done = env.step(action)

        print(f"Total Score: {total_score}, Max Num: {max_num}")
        env.render_board()