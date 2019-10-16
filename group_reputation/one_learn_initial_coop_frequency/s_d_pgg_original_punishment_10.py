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
            for j in range(i * self.g_s, (i + 1) * self.g_s):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(g_s, g_b, g_l):
    t_n = g_s * (g_b ** (g_l - 1))
    s_s = SocialStructure(g_s, g_b, g_l, t_n)
    ind_pos, pos_ind = s_s.build_social_structure()
    return ind_pos, pos_ind


def initialize_strategy(t_n):
    # 0 for defect, 1 for cooperate, 2 for cooperate and punish
    init_stra = np.random.choice([0, 1, 2], t_n, p = [1./2., 1./4., 1./4.])
    return init_stra


def game_one_round(stra_l, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    stra_l_old = stra_l[:]
    p_l = [0 for _ in range(ind_n)]
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
            elif stra_l[g_inds[i]] == 2: # strategy == 2 -- punish
                g_a.append(1)
                stra_count[2] += 1
        g_p = pgg_game(g_a, gamma)
        for i in range(g_inds_n):
            p_l[g_inds[i]] += g_p[i]
        if stra_count[0] > 0:
            for i in range(g_inds_n):
                if stra_l[g_inds[i]] == 2:
                    p_l[g_inds[i]] -= 1.0
                if stra_l[g_inds[i]] == 0:
                    p_l[g_inds[i]] -= 1.0 * stra_count[2] / stra_count[0]
    for _ in range(pos_n):
        ind = random.choice(pos_ind[_])
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
            ind_p = p_l[ind]
            oppon_p = p_l[oppon]
            t1 = 1 / (1 + math.e ** (10 * (ind_p - oppon_p)))
            t2 = random.random()
            if t2 < t1:
                stra_l[ind] = stra_l_old[oppon]
    return stra_l

def run_game(run_time, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    stra_l = initialize_strategy(ind_n)
    for step in range(run_time):
        stra_l = game_one_round(stra_l, gamma, ind_pos, pos_ind)
    return stra_l

def evaluation(eval_time, gamma, ind_pos, pos_ind, stra_l):
    ind_n = len(ind_pos)
    stra_frac = np.array([0, 0, 0])
    for step in range(eval_time):
        new_stra_frac = np.array([0, 0, 0])
        stra_l = game_one_round(stra_l, gamma, ind_pos, pos_ind)
        for i in stra_l:
           new_stra_frac[i] += 1
        new_stra_frac = new_stra_frac / ind_n
        stra_frac = step / (step + 1) * stra_frac + 1 / (step + 1) * new_stra_frac

    return stra_frac


if __name__ == '__main__':
    group_size_r = 16; group_base_r = 2; group_length_r = 5
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    run_time = 1000; eval_time = 200
    init_time = 50
    result_stra_frac = np.array([0, 0, 0])
    result = {}
    for gamma_r in np.arange(0.1, 1.3, 0.1):
        gamma_r = round(gamma_r, 2)
        print(gamma_r)
        for i in range(init_time):
            stra_l_r = run_game(run_time, gamma_r, ind_pos_r, pos_ind_r)
            stra_frac_r = evaluation(eval_time, gamma_r, ind_pos_r, pos_ind_r, stra_l_r)
            result_stra_frac = i / (i + 1) * result_stra_frac + 1 / (i + 1) * stra_frac_r
        result[gamma_r] = result_stra_frac
    result = pd.DataFrame(result).T
    result.to_csv('./results/s_d_pgg_original_punishment_10.csv')
    print(result)