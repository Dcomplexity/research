from game_env import *
import s_a_dist as sad
import s_pi_dist as spd
import os
import pandas as pd
import datetime
import numpy as np
import itertools
import argparse

from multiprocessing import Pool


def valid_s(s_value):
    if s_value < 0.001:
        s_new = 0.001
    elif s_value > 0.999:
        s_new = 0.999
    else:
        s_new = s_value
    return s_new


def calc_payoff(agent_id, s, a_l, mixed_s, p_m, pi):
    p = 0
    if agent_id == 0:
        for act_i in a_l:
            p_j = 0
            for act_j in a_l:
                p_j += p_m[s * 4 + act_i * 2 + act_j][0] * pi[1 - agent_id][s][act_j]
            p += mixed_s[act_i] * p_j
    else:
        for act_i in a_l:
            p_j = 0
            for act_j in a_l:
                p_j += p_m[s * 4 + act_j * 2 + act_i][1] * pi[1 - agent_id][s][act_j]
            p += mixed_s[act_i] * p_j
    return p


def evolve(strategy, step_size, transition_matrix):
    s_l = [0, 1]
    a_l = [0, 1]
    s00, s01, s10, s11 = strategy
    pi = [{0: [1 - s00, s00], 1: [1 - s01, s01]}, {0: [1 - s10, s10], 1: [1 - s11, s11]}]
    s_dist, p_matrix = sad.run(pi, transition_matrix)
    s_pi_dist = spd.gen_s_pi_dist(s_l, a_l, pi, transition_matrix)
    ds00 = (calc_payoff(0, 0, a_l, [0, 1], p_matrix, pi) - calc_payoff(0, 0, a_l, pi[0][0], p_matrix, pi)) * s00 * \
           s_pi_dist[0]
    ds01 = (calc_payoff(0, 1, a_l, [0, 1], p_matrix, pi) - calc_payoff(0, 1, a_l, pi[0][1], p_matrix, pi)) * s01 * \
           s_pi_dist[1]
    ds10 = (calc_payoff(1, 0, a_l, [0, 1], p_matrix, pi) - calc_payoff(1, 0, a_l, pi[1][0], p_matrix, pi)) * s10 * \
           s_pi_dist[0]
    ds11 = (calc_payoff(1, 1, a_l, [0, 1], p_matrix, pi) - calc_payoff(1, 1, a_l, pi[1][1], p_matrix, pi)) * s11 * \
           s_pi_dist[1]
    # s00 = valid_s(s00 + ds00 * step_size)
    # s01 = valid_s(s01 + ds01 * step_size)
    # s10 = valid_s(s10 + ds10 * step_size)
    # s11 = valid_s(s11 + ds11 * step_size)
    # return [s00, s01, s10, s11]
    return [ds00, ds01, ds10, ds11]


# def run_task(p_init):
def run_task():
    states = [0, 1]
    actions = [0, 1]
    # t = np.arange(0, 1000000)
    step_length = 0.001
    # print(p_init)
    for p_1, p_2 in [[0.9, 0.1], [0.5, 0.5]]:
        print(p_1, p_2)
        transition_matrix = [[p_1, p_1, p_1, p_2], [p_2, p_2, p_2, p_1]]
        p_sub = np.round(np.arange(0.1, 1.0, 0.1), 2)
        p_prod_s1 = list(itertools.product(p_sub, p_sub))
        p_prod = np.ones((len(p_prod_s1), 4)) * 0.01
        for i in range(len(p_prod_s1)):
            p_prod[i][0] = p_prod_s1[i][0]
            p_prod[i][2] = p_prod_s1[i][1]
        ds_list = []
        for p_init in p_prod:
            print(p_init)
            # if _ % 1000 == 0:
            #     print(_)
            ds = evolve(p_init, step_length, transition_matrix)
            # if _ % 1000 == 0:
            #     print(p)
            ds_list.append(ds)
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results"))
        csv_file_name = "/p1_%.1f_p2_%.1f_strategy_trace.csv" % (p_1, p_2)
        file_name = abs_path + csv_file_name
        d_pd = pd.DataFrame(ds_list)
        d_pd.to_csv(file_name, index=None)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p1', '--p1', type=float, default=0.9, help="transition probability")
    # parser.add_argument('-p2', '--p2', type=float, default=0.1, help="transition probability")
    # args = parser.parse_args()


    # transition_matrix = [[0.1, 0.1, 0.1, 0.9], [0.9, 0.9, 0.9, 0.1]]
    # transition_matrix = [[0.9, 0.9, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]
    # transition_matrix = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    # p_1 = args.p1
    # p_2 = args.p2
    # print(p_1, p_2)


    run_task()

