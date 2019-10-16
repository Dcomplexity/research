import pandas as pd
import random
import math

from frac_time.game import *

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


def initial_action(t_n):
    init_a = np.random.choice([0, 1], t_n, p = [0.5, 0.5])
    return init_a


def game_one_round(a_l, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = a_l[:]
    p_l = [0 for _ in range(ind_n)]
    for pos in range(pos_n):
        g_inds = pos_ind[pos]
        g_inds_n = len(g_inds)
        g_a = []
        for i in range(g_inds_n):
            g_a.append(a_l[g_inds[i]])
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
    return a_l, p_l


def run_game_frac_time(run_time, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    act_history = []
    payoff_history = []
    act_l = initial_action(ind_n)
    act_history.append(np.copy(act_l[:]))
    for step in range(run_time):
        act_l, p_l = game_one_round(act_l, gamma, ind_pos, pos_ind)
        act_history.append(np.copy(act_l[:]))
        payoff_history.append(np.copy(p_l[:]))
    act_l, p_l = game_one_round(act_l, gamma, ind_pos, pos_ind)
    payoff_history.append(np.copy(p_l[:]))
    return act_history, payoff_history


if __name__ == '__main__':
    group_size_r = 16; group_base_r = 2; group_length_r = 5
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    run_time = 1000
    result_act = []
    result_payoff = []
    gamma_l = []
    for i in np.arange(0.1, 1.3, 0.1):
        gamma_l.append(round(i, 2))
    for gamma_r in gamma_l:
        print(gamma_r)
        act_history_r, payoff_history_r = run_game_frac_time(run_time, gamma_r, ind_pos_r, pos_ind_r)
        result_act.extend(act_history_r)
        result_payoff.extend(payoff_history_r)
    step_l = np.arange(run_time + 1)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    result_act_pd = pd.DataFrame(result_act, index=m_index)
    result_payoff_pd = pd.DataFrame(result_payoff, index=m_index)
    result_act_pd.to_csv('./results/ft_pgg_original_act.csv')
    result_payoff_pd.to_csv('./results/ft_pgg_original_payoff.csv')
    print(result_act_pd)
    print(result_payoff_pd)


