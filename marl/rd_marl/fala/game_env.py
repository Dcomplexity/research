import numpy as np
import random

# pd_game_1 = [[0.2, 0.2], [1, 0], [0, 1], [0.3, 0.3]]
# pd_game_2 = [[0.1, 0.1], [1, 0], [0, 1], [0.4, 0.4]]
pd_game_1 = [[0, 0], [2, -1], [-1, 2], [1, 1]]
pd_game_2 = [[0, 0], [1.2, -1], [-1, 1.2], [0.2, 0.2]]
# The probability to next state based on the joint action
# transition_matrix = [[0.1, 0.8, 0.9, 0.1], [0.1, 0.9, 0.9, 0.1]]
transition_matrix = [[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]

def play_pd_game_1(a_x, a_y):
    return pd_game_1[a_x * 2 + a_y]

def play_pd_game_2(a_x, a_y):
    return pd_game_2[a_x * 2 + a_y]


def transition_prob(s, s_, a_x, a_y):
    if s != s_:
        return transition_matrix[s][a_x * 2 + a_y]
    else:
        return 1 - transition_matrix[s][a_x * 2 + a_y]

def next_state(s, a_x, a_y):
    prob = transition_matrix[s][a_x * 2 + a_y]
    if random.random() < prob:
        s_ = 1 - s
    else:
        s_ = s
    return s_


if __name__ == "__main__":
    s_sum = 0
    for i in range(1000):
        p_s = next_state(1, 1, 1)
        s_sum += p_s
    print(s_sum)