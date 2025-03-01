import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np

from env.board_2048 import Board_2048

class RandomAgent:
    def __init__(self):
        self.action_size = 4

    def get_action(self, state):
        return np.random.choice(self.action_size)

if __name__ == "__main__":
    env = Board_2048()
    agent = RandomAgent()

    episodes = 1000
    total_score_history, max_num_history = [], []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, done = env.step(action)

        total_score, max_num = env.get_scores()

        total_score_history.append(total_score)
        max_num_history.append(max_num)
        if episode % 100 == 0:
            print("episode: {}, total score: {}, max number: {}".format(episode, total_score, max_num))

    plt.subplot(211)
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    plt.plot(range(len(total_score_history)), total_score_history)

    plt.subplot(212)
    plt.xlabel('Episode')
    plt.ylabel('Max Number')
    plt.plot(range(len(max_num_history)), max_num_history)

    plt.show()