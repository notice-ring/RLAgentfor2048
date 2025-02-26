import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import random
import copy

class Board_2048():
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "LEFT",
            1: "UP",
            2: "RIGHT",
            3: "DOWN",
        }

        # 4 * 4 matrix
        self.board= np.zeros((4, 4), dtype=int)

        # start with random position of two '2'
        for _ in range(2):
            self.add_rnd_num()

        self.done = False
        self.total_score = 0
        self.plus_score = 0

    def reset(self):  # reset the board
        self.__init__()

        return self.board

    # add 2 or 4 on randome position of the board
    def add_rnd_num(self):
        h, w = (self.board == 0).nonzero()

        if h.size != 0:
            rnd_idx = random.randint(0, h.size - 1)
            self.board[h[rnd_idx], w[rnd_idx]] = 2 * ((random.random() > .9) + 1)

    @property
    def height(self):
        return self.board.shape[0]

    @property
    def width(self):
        return self.board.shape[1]

    @property
    def shape(self):
        return self.board.shape

    def actions(self):  # return all actions
        return self.action_space

    def states(self):  # return all states
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def move_left(self, next_board):  # move elements to left
        self.plus_score = 0

        for h in range(self.height):
            elements = next_board[h][np.nonzero(next_board[h])]

            w = 0
            while w < len(elements) - 1:
                if elements[w] == elements[w+1]:
                    elements[w] *= 2
                    self.plus_score += elements[w]
                    elements = np.delete(elements, w+1)

                w += 1

            next_board[h] = np.concatenate([elements, np.zeros(self.width - len(elements))])

        return next_board

    def move_action(self, action):  # call functions move_up, move_down, move_left, move_right
        new_board = copy.deepcopy(self.board)
        roatated_board = np.rot90(new_board, action)
        next_board = self.move_left(roatated_board)

        return np.rot90(next_board, -action)


    def is_action(self, action):  # judge if action is valid or not
        next_board = self.move_action(action)

        return not (next_board == self.board).all()


    def next_state(self, action):  # move and add random number on the board, add score to total score  
        next_board = self.move_action(action)
        self.board = next_board
        self.total_score += self.plus_score

        self.add_rnd_num()


    def reward(self, action, next_board):  # return reward, which is plus score
        return self.plus_score

    def is_done(self):  # return bool of the state that board can't move to anyway
        for h in range(self.height):
            for w in range(self.width):
                if self.board[h][w] == 0:
                    return False
                if h != 0 and self.board[h-1][w] == self.board[h][w]:
                    return False
                if w != 0 and self.board[h][w-1] == self.board[h][w]:
                    return False

        return True

    def step(self, action):  # return board, reward, done after action
        self.next_state(action)

        reward = self.reward(action, self.board)

        done = self.is_done()

        return self.board, reward, done

    def get_scores(self):  # return total_score and max number of board when game ends
        return self.total_score, self.board.max()

    def render_board(self):  # render the board
        for h in range(self.height):
            for w in range(self.width):
                print("{0:4d}".format(self.board[h][w]), end=" ")
            print()

        print("score : ", self.total_score)


if __name__ == "__main__":
    env = Board_2048()
    done = False

    while not done:
        env.render_board()
        action = input("(0):Left, (1):UP (2):RIGHT (3):DOWN\n")

        if action.isdigit():
            action = int(action)
        else:
            print("Enter a number!")
            continue

        if action < 0 or action > 3:
            print("Not a valid action! Enter a valid action (0/1/2/3)")
            continue

        is_action = env.is_action(action)
        if not is_action:
            print("Not a valid action! Nothing has changed.")
            continue
        else:
            _, _, done = env.step(action)

    env.render_board()
    print("Game Over! Total Score:{} Max Num:{}".format(env.get_scores()))