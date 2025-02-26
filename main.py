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
from agent.DQNAgent import DQNAgent

def train():
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

            torch.save(agent.qnet.state_dict(), '/content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet.pth')
            torch.save(agent.qnet_target.state_dict(), '/content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet_target.pth')

    torch.save(agent.qnet.state_dict(), '/content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet.pth')
    torch.save(agent.qnet_target.state_dict(), '/content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet_target.pth')

    plt.subplot(211)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.plot(range(len(total_score_history)), total_score_history)

    plt.subplot(212)
    plt.xlabel('Episode')
    plt.ylabel('Max Number')
    plt.plot(range(len(max_num_history)), max_num_history)

    plt.show()

def test():
    env = Board_2048()
    agent = DQNAgent()
    with torch.no_grad():
        agent.qnet.load_state_dict(torch.load('content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet.pth'))
        agent.qnet_target.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/rl_2048/model/qnet_target.pth'))

        agent.qnet.eval()
        agent.qnet_target.eval()

        state = env.reset()
        done = False

        while not done:
            action = agent.get_action_exploitation(state)
            is_action = env.is_action(action)
            if not is_action:
                continue
            else:
                state, reward, done = env.step(action)

        total_score, max_num = env.get_scores()

        print(f"Total Score: {total_score}, Max Num: {max_num}")
        env.render_board()


if __name__ == '__main__':
    train()
    test()