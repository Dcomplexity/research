import numpy as np
from scipy.linalg import solve
from game_env import *

def q_z_s(z, s, a_l, pi):
    q = 0
    for a_x in a_l:
        for a_y in a_l:
            q += transition_prob(z, s, a_x, a_y) * pi[0][z][a_x] * pi[1][z][a_y]
    return q


def q_matrix(s_l, a_l, pi):
    q_m = []
    for z in s_l:
        q_m.append([])
        for s in s_l:
            q_m[z].append(q_z_s(z, s, a_l, pi))
    return q_m


def gen_s_pi_dist(s_l, a_l, pi):
    q_m = q_matrix(s_l, a_l, pi)
    # a = np.array([[q_m[0][0] - 1, q_m[1][0]], [1, 1]])
    a = np.array([[q_m[0][1], q_m[1][1] - 1], [1, 1]])
    b = np.array([0, 1])
    x = solve(a, b)
    # print(x)
    # x = x / np.sum(x)
    return x

if __name__ == '__main__':
    # the first dict for player 0 and the second dict for player 1,
    # In the dict, 0 for state 0 and 1 for state 1
    policy_pi = [{0: [0.5, 0.5], 1: [0.5, 0.5]}, {0: [0.5, 0.5], 1: [0.5, 0.5]}]
    states = [0, 1]
    actions = [0, 1]
    state_dist = gen_s_pi_dist(states, actions, policy_pi)
    print(state_dist)