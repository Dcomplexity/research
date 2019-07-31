import numpy as np
import pandas as pd
import random
from game_env import *


def evolution_one_step(total_num, r, strategy):
    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        if random.random() < strategy:
            a_l[i] = 1
    pf_l = play_pgg_game(a_l, r)
    num_c = 0
    num_d = 0
    p_c = 0
    p_d = 0
    for i in range(total_num):
        if a_l[i] == 1:
            num_c += 1
            p_c += pf_l[i]
        else:
            num_d += 1
            p_d += pf_l[i]
    print(p_c, p_d)
    p_c_ave = p_c / (num_c + 0.0001)
    p_d_ave = p_d / (num_d + 0.0001)
    print(p_c_ave, p_d_ave)
    strategy_gradient = strategy * (1 - strategy) * (p_c_ave - p_d_ave)
    strategy = strategy + 0.01 * strategy_gradient
    return strategy


def run(run_time):
    strategy = 0.7
    total_num = 100
    r = strategy * total_num
    for _ in range(run_time):
        strategy = evolution_one_step(total_num, r, strategy)
        print(strategy)

if __name__ == '__main__':
    run(1000)




