import numpy as np
import random


def generate_chess_board(xdim, ydim, prob):
    c_b = np.zeros((xdim, ydim))
    for i in range(c_b.shape[0]):
        for j in range(c_b.shape[1]):
            if random.random() < prob:  # prob is the probability that the element is white
                c_b[i, j] = 1
    return c_b


class Agent:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.trace = []
        self.detect_color_frac = 0.5
        self.decision = None

    def move(self, direction, boundary):
        """
        Agent move one step
        :param direction: a list, the agent move direction, [x_axis, y_axis], one of them is -1 or 1
        :param boundary: [[x_left_b, x_right_b], [y_low_b, y_high_b]]
        :return: new position
        """
        self.pos = self.pos + direction
        if self.pos[0] < boundary[0][0]:
            self.pos[0] = boundary[0][0]
        elif self.pos[0] > boundary[0][1]:
            self.pos[0] = boundary[0][1]
        if self.pos[1] < boundary[1][0]:
            self.pos[1] = boundary[1][0]
        elif self.pos[1] > boundary[1][1]:
            self.pos[1] = boundary[1][1]

    def detect(self, board, pos):
        color = board[pos[0], pos[1]]
        self.trace.append(color)

    def make_decision(self):
        if self.trace == []:
            print("None Sense")
        else:
            self.color_frac_sense = np.mean(self.trace)
        if self.detect_color_frac < 0.5:
            self.decision = 0
        elif self.color_frac_sense == 0.5:
            self.decision = np.random.choice([0, 1])
        else:
            self.decision = 1

    def get_decision(self):
        return self.decision



if __name__ == '__main__':
    chess_board = generate_chess_board(20, 20, 0.4)

