import pandas as pd
import random
import math

from average_initial_frequency.game import *

class SocialStructure():
    def __init__(self, g_s, g_b, g_l, t_n):
        self.g_s = g_s  # group_size
        self.g_b = g_b  # group_base
        self.g_l = g_l  # group_length
        self.t_n = t_n  # total_num
        self.g_n = self.g_b ** (self.g_l - 1)
        if self.t_n != self.g_s * self.g_n:
            print("Error: The total num of individuals does not correspond to the social structure")

    def build_social_structure(self):
        ind_pos = [0 for x in range(self.t_n)]
        pos_ind = [[] for x in range(self.g_n)]
        for i in range(self.g_n):
            for j in range(i*self.g_s, (i+1)*self.g_s):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(g_s, g_b, g_l):
    t_n = g_s * (g_b ** (g_l - 1))
    s_s = SocialStructure(g_s, g_b, g_l, t_n)
    ind_pos, pos_ind = s_s.build_social_structure()
    return ind_pos, pos_ind


def initialize_action(t_n):
    init_a = np.random.choice([0, 1], t_n, p = [0.5, 0.5])
    return init_a


def initialize_gamma(pos_n, init_gamma_value):
    init_gamma = [init_gamma_value for _ in range(pos_n)]
    return init_gamma


def build_local_neigh(g_b, g_l, distance):
    pos_num = (g_b ** (g_l - 1))
    all_pos = np.arange(pos_num)
    poten_pos = []
    if distance == 1:
        for i in range(pos_num):
            poten_pos.append([i])
    else:
        poten_num = 2 ** (distance - 1) ## group_length > distance >= 2
        poten_pos_num = 2 ** (g_l - distance)
        for k in range(0, poten_pos_num):
            poten_pos.append(all_pos[k * poten_num: (k + 1) * poten_num])
    return poten_pos


def game_one_round(a_l, gamma_l, ind_pos, pos_ind, ave_gamma):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = a_l[:]
    p_l = [0 for _ in range(ind_n)]
    g_a_frac = [0 for _ in range(pos_n)]
    for pos in range(pos_n):
        gamma = gamma_l[pos]
        g_inds = pos_ind[pos]
        g_inds_n = len(g_inds)
        g_a = []
        for i in range(g_inds_n):
            g_a.append(a_l[g_inds[i]])
        g_a_frac[pos] = np.mean(g_a)
        g_p = pgg_game(g_a, gamma)
        for i in range(g_inds_n):
            p_l[g_inds[i]] += g_p[i]
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
            ind_p = p_l[ind]
            oppon_p = p_l[oppon]
            t1 = 1 / (1 + math.e ** (10 * (ind_p - oppon_p)))
            t2 = random.random()
            if t2 < t1:
                a_l[ind] = a_l_old[oppon]

    gamma_group = build_local_neigh(g_b = 2, g_l =5, distance=2)
    # Update gamma_l
    for _ in range(len(gamma_group)):
        total_a_frac = 0
        neigh_length = len(gamma_group[0])
        for pos in gamma_group[_]:
            total_a_frac += g_a_frac[pos]
        for pos in gamma_group[_]:
            gamma_l[pos] = ave_gamma * neigh_length * (g_a_frac[pos] + 0.001) / (total_a_frac + 0.001 * neigh_length)

    return a_l, gamma_l


def run_game(run_time, ave_gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l = initialize_action(ind_n)
    gamma_l = initialize_gamma(pos_n, ave_gamma)
    for step in range(run_time):
        a_l, gamma_l = game_one_round(a_l, gamma_l, ind_pos, pos_ind, ave_gamma)
    return a_l, gamma_l


def evaluation(eval_time, ave_gamma, ind_pos, pos_ind, a_l, gamma_l):
    ind_n = len(ind_pos)
    a_frac = 0
    for step in range(eval_time):
        a_l, gamma_l = game_one_round(a_l, gamma_l, ind_pos, pos_ind, ave_gamma)
        a_frac = step / (step + 1) * a_frac + 1 / (step + 1) * np.mean(a_l)
    return a_frac


if __name__ == '__main__':
    group_size_r = 16; group_base_r = 2; group_length_r = 5
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    run_time = 100; eval_time = 20
    init_time = 10
    result_a_frac = 0
    result = {}
    for gamma_r in np.arange(0.1, 1.3, 0.1):
        ave_gamma_r = round(gamma_r, 2)
        print(ave_gamma_r)
        for i in range(init_time):
            a_l_r, gamma_l_r = run_game(run_time, ave_gamma_r, ind_pos_r, pos_ind_r)
            a_frac_r = evaluation(eval_time, ave_gamma_r, ind_pos_r, pos_ind_r, a_l_r, gamma_l_r)
            result_a_frac = i / (i + 1) * result_a_frac + 1 / (i + 1) * a_frac_r
        result[ave_gamma_r] = [result_a_frac]
    result = pd.DataFrame(result).T
    result.to_csv('./results_old/s_d_pgg_competitive_local_gamma.csv')
    print(result)