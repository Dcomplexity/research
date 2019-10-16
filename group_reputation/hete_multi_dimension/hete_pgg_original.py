import pandas as pd
import random
import math

from hete_multi_dimension.game import *
from hete_multi_dimension.social_structure import *

def initial_action(t_n):
    init_a = np.random.choice([0, 1], t_n, p=[0.5, 0.5])
    return init_a


def game_one_round(multi_dimen, a_l, gamma, ind_pos_multi, pos_ind_multi):
    ind_n = len(a_l)
    p_all_l = np.zeros(ind_n)
    a_l_old = a_l[:]
    for d in range(multi_dimen):
        ind_pos = ind_pos_multi[d]
        pos_ind = pos_ind_multi[d]
        pos_n = len(pos_ind)
        p_l = np.zeros(ind_n)
        for pos in range(pos_n):
            g_inds = pos_ind[pos]
            g_inds_n = len(g_inds)
            g_a = []
            for i in range(g_inds_n):
                g_a.append(a_l[g_inds[i]])
            g_p = pgg_game(g_a, gamma)
            for i in range(g_inds_n):
                p_l[g_inds[i]] += g_p[i]
        p_all_l = (d / (d + 1)) * p_all_l + (1 / (d + 1)) * p_l
    for ind in range(ind_n):
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:  # mutation
            a_l[ind] = 1 - a_l[ind]
        else:
            while True:
                oppon = random.choice(range(ind_n))
                if oppon != ind:
                    break
            ind_p = p_all_l[ind]
            oppon_p = p_all_l[oppon]
            t1 = 1 / (1 + math.e ** (10 * (ind_p - oppon_p)))
            t2 = random.random()
            if t2 < t1:
                a_l[ind] = a_l_old[oppon]
    return a_l


def run_game(run_time, multi_dimen, gamma, ind_pos_multi, pos_ind_multi):
    ind_n = len(ind_pos_multi[0])
    a_l = initial_action(ind_n)
    for step in range(run_time):
        a_l = game_one_round(multi_dimen, a_l, gamma, ind_pos_multi, pos_ind_multi)
    return a_l


def evaluation(eval_time, multi_dimen, gamma, ind_pos_multi, pos_ind_multi, a_l):
    ind_n = len(ind_pos_multi[0])
    a_frac = 0
    for step in range(eval_time):
        a_l = game_one_round(multi_dimen, a_l, gamma, ind_pos_multi, pos_ind_multi)
        a_frac = step / (step + 1) * a_frac + 1 / (step + 1) * np.mean(a_l)
    return a_frac


if __name__ == '__main__':
    group_size_r = 16; group_base_r = 2; group_length_r = 5; multi_dimen_r = 5
    total_number_r = group_size_r * (group_base_r ** (group_length_r - 1))
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    pos_ind_multi_r = build_hete_pos_ind(multi_dimen_r, pos_ind_r, group_length_r, group_size_r)
    ind_pos_multi_r = build_hete_ind_pos(pos_ind_multi_r, total_number_r)
    run_time = 1000; eval_time = 200
    init_time = 10
    result_a_frac = 0
    result = {}
    for gamma_r in np.arange(0.1, 1.3, 0.1):
        gamma_r = round(gamma_r, 2)
        print(gamma_r)
        for i in range(init_time):
            a_l_r= run_game(run_time, multi_dimen_r, gamma_r, ind_pos_multi_r, pos_ind_multi_r)
            a_frac_r = evaluation(eval_time, multi_dimen_r, gamma_r, ind_pos_multi_r, pos_ind_multi_r, a_l_r)
            result_a_frac = i / (i + 1) * result_a_frac + 1 / (i + 1) * a_frac_r
        result[gamma_r] = [result_a_frac]
    result = pd.DataFrame(result).T
    result_file_name = './results/hete_pgg_original.csv'
    result.to_csv(result_file_name)
    print(result)