import pandas as pd
import random
import math

from multi_dimension.game import *
from multi_dimension.social_structure import *

def initialize_strategy(t_n):
    # 0 for defect, 1 for cooperate, 2 for cooperate and punish
    init_stra = np.random.choice([0, 1, 2], t_n, p = [1./2., 1./4., 1./4.])
    return init_stra


def game_one_round(multi_dimen, stra_l, gamma, ind_pos_multi, pos_ind_multi):
    ind_n = len(stra_l)
    p_all_l = np.zeros(ind_n)
    stra_l_old = stra_l[:]
    for d in range(multi_dimen):
        ind_pos = ind_pos_multi[d]
        pos_ind = pos_ind_multi[d]
        pos_n = len(pos_ind)
        p_l = np.zeros(ind_n)
        for pos in range(pos_n):
            g_inds = pos_ind[pos]
            g_inds_n = len(g_inds)
            g_a = []
            stra_count = [0, 0, 0]
            for i in range(g_inds_n):
                if stra_l[g_inds[i]] == 0:
                    g_a.append(0)
                    stra_count[0] += 1
                elif stra_l[g_inds[i]] == 1:
                    g_a.append(1)
                    stra_count[1] += 1
                elif stra_l[g_inds[i]] == 2:  # strategy == 2 -- punish
                    g_a.append(1)
                    stra_count[2] += 1
            g_p = pgg_game(g_a, gamma)
            for i in range(g_inds_n):
                p_l[g_inds[i]] += g_p[i]
            if stra_count[0] > 0:
                for i in range(g_inds_n):
                    if stra_l[g_inds[i]] == 2:
                        p_l[g_inds[i]] -= 0.3
                    if stra_l[g_inds[i]] == 0:
                        p_l[g_inds[i]] -= 1.0 * stra_count[2] / stra_count[0]
        p_all_l = d / (d + 1) * p_all_l + 1 / (d + 1) * p_l

    for ind in range(ind_n):
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:  # mutation
            p_stra = [0, 1, 2]
            p_stra.remove(stra_l_old[ind])
            stra_l[ind] = random.choice(p_stra)
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
                stra_l[ind] = stra_l_old[oppon]
    return stra_l


def run_game(run_time, multi_dimen, gamma, ind_pos_multi, pos_ind_multi):
    ind_n = len(ind_pos_multi[0])
    stra_l = initialize_strategy(ind_n)
    for step in range(run_time):
        stra_l = game_one_round(multi_dimen, stra_l, gamma, ind_pos_multi, pos_ind_multi)
    return stra_l


def evaluation(eval_time, multi_dimen, gamma, ind_pos_multi, pos_ind_multi, stra_l):
    ind_n =len(ind_pos_multi[0])
    stra_frac = np.array([0, 0, 0])
    for step in range(eval_time):
        new_stra_frac = np.array([0, 0, 0])
        stra_l = game_one_round(multi_dimen, stra_l, gamma, ind_pos_multi, pos_ind_multi)
        for i in stra_l:
            new_stra_frac[i] += 1
        new_stra_frac = new_stra_frac / ind_n
        stra_frac = step / (step + 1) * stra_frac + 1 / (step + 1) * new_stra_frac
    return stra_frac


if __name__ == '__main__':
    group_size_r = 16; group_base_r = 2; group_length_r = 5; multi_dimen_r = 5; beta_r = -5
    total_number_r = group_size_r * (group_base_r ** (group_length_r - 1))
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    pos_ind_multi_r = build_multi_pos_ind(multi_dimen_r, pos_ind_r, group_length_r, group_size_r, beta_r)
    ind_pos_multi_r = build_multi_ind_pos(pos_ind_multi_r, total_number_r)
    run_time = 1000; eval_time = 200
    init_time = 10
    result_stra_frac = 0
    result = {}
    for gamma_r in np.arange(0.1, 1.3, 0.1):
        gamma_r = round(gamma_r, 2)
        print(gamma_r)
        for i in range(init_time):
            stra_l_r = run_game(run_time, multi_dimen_r, gamma_r, ind_pos_multi_r, pos_ind_multi_r)
            stra_frac_r = evaluation(eval_time, multi_dimen_r, gamma_r, ind_pos_multi_r, pos_ind_multi_r, stra_l_r)
            result_stra_frac = i / (i + 1) * result_stra_frac + 1 / (i + 1) * stra_frac_r
        result[gamma_r] = result_stra_frac
    result = pd.DataFrame(result).T
    result_file_name = './results/sd_pgg_original_punishment_03_%d.csv' %(beta_r)
    result.to_csv(result_file_name)
    print(result)


