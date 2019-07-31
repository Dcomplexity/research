import numpy as np
import random
import math
import pandas as pd
import os

def play_donation_game(a_x, a_y, b):
    if a_x ==  1 and a_y == 1:
        return b-1, b-1
    elif a_x == 1 and a_y == 0:
        return -1, b
    elif a_x == 0 and a_y == 1:
        return b, -1
    elif a_x == 0 and a_y == 0:
        return 0, 0
    else:
        return "Error"


def play_b_game(a_x, a_y, b):
    if a_x == 1 and a_y == 1:
        return 1, 1
    elif a_x == 1 and a_y == 0:
        return 0, b
    elif a_x == 0 and a_y == 1:
        return b, 0
    elif a_x == 0 and a_y == 0:
        return 0, 0
    else:
        return "Error"


def play_donation_c_game(a_x, a_y, b, c):
    if a_x == 1 and a_y == 1:
        return b-c, b-c
    elif a_x == 1 and a_y == 0:
        return -c, b
    elif a_x == 0 and a_y == 1:
        return b, -c
    elif a_x == 0 and a_y == 0:
        return 0, 0
    else:
        return "Error"


def play_pd_game(a_x, a_y, r, s, t, p):
    if a_x == 1 and a_y == 1:
        return r, r
    elif a_x == 1 and a_y == 0:
        return s, t
    elif a_x == 0 and a_y == 1:
        return t, s
    elif a_x == 0 and a_y == 0:
        return p, p
    else:
        return "Error"


def play_pgg_game(a_l, r):
    agent_num = len(a_l)
    pf = np.sum(a_l) * r / agent_num
    pf_l = np.array([pf for _ in range(agent_num)]) - np.array(a_l)
    return pf_l

if __name__ == '__main__':
    pf_l_r = play_pgg_game([0, 1, 1], 2.0)
    print(pf_l_r)








