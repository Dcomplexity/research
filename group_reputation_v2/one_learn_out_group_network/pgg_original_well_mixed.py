import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def pgg_game(a_l, gamma):  # r in (0, 1]
    a_n = len(a_l)
    p = np.array([np.sum(a_l) * gamma * a_n / a_n for _ in range(a_n)]) - np.array(a_l)
    return p


class SocialStructure():
    def __init__(self, g_s, g_n, t_n):
        self.g_s = g_s  # group_size
        self.g_n = g_n  # group_num
        self.t_n = t_n  # total_num
        if self.t_n != self.g_s * self.g_n:
            print ("Error: The total num of individuals does not correspond to the social structure")

    def build_social_structure(self):
        ind_pos = [0 for x in range(self.t_n)]
        pos_ind = [[] for x in range(self.g_n)]
        for i in range(self.g_n):
            for j in range(i*self.g_s, (i+1)*self.g_s):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(g_s, g_n):
    t_n = g_s * g_n
    s_s = SocialStructure(g_s, g_n, t_n)
    ind_pos, pos_ind = s_s.build_social_structure()
    return ind_pos, pos_ind


def initial_action(f_0, ind_pos, pos_ind):
    init_a = []
    g_n = len(pos_ind)
    for pos in range(g_n):
        group_ind = pos_ind[pos]
        group_ind_n = len(group_ind)
        group_ind_a = np.random.choice([0, 1], group_ind_n, p = [1 - f_0[pos], f_0[pos]])
        init_a.append(group_ind_a)
    return np.array(init_a)


def game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w, mu):
    ind_n = len(ind_pos)
    all_ind = np.arange(ind_n)
    np.random.shuffle(all_ind)
    pos_n = len(pos_ind)
    new_ind_pos = [0 for _ in range(ind_n)]
    new_pos_ind = [[] for _ in range(pos_n)]
    for i in range(pos_n):
        for j in range(i * g_s, (i + 1) * g_s):
            new_ind_pos[all_ind[j]] = i
            new_pos_ind[i].append(all_ind[j])
    a_l_old = np.copy(a_l)
    ind_a_l = a_l.flatten()
    ind_a_l_old = a_l_old.flatten()
    p_l = [0 for _ in range(ind_n)]
    for pos in range(pos_n):
        g_a = []
        for j in new_pos_ind[pos]:
            g_a.append(ind_a_l[j])
        g_p = pgg_game(g_a, gamma)
        pos_i_n = 0
        for j in new_pos_ind[pos]:
            p_l[j] = g_p[[pos_i_n]]
            pos_i_n += 1
    ind_p_l = np.array(p_l)
    for pos in range(pos_n):
        if random.random() < mu:
            g_ind = pos_ind[pos]
            ind = random.choice(g_ind)
            ind_a_l[ind] = 1 - ind_a_l_old[ind]
        else:
            g_ind = pos_ind[pos]
            ind = random.choice(g_ind)
            while True:
                oppon = random.choice(range(ind_n))
                if oppon not in g_ind:
                    break
            ind_p = ind_p_l[ind]
            oppon_p = ind_p_l[oppon]
            t1 = 1 / (1 + math.e ** (2.0 * (ind_p - oppon_p)))
        # t1 = (1 / 2 + w / (2 * delta_pi) * (oppon_p - ind_p))
            t2 = random.random()
            if t2 < t1:
                ind_a_l[ind] = ind_a_l_old[oppon]
    return ind_a_l


def run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w, mu):
    f_history = []
    for round in range(init_time):
        a_l = initial_action(f_0, ind_pos, pos_ind)
        for step in range(run_time):
            if round == 0:
                f_history.append(a_l.mean())
            else:
                f_history[step] = round / (round + 1) * f_history[step] + 1 / (round + 1) * a_l.mean()
            a_l = game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w, mu)
        if round == 0:
            f_history.append(a_l.mean())
        else:
            f_history[run_time] = round / (round + 1) * f_history[run_time] + 1 / (round + 1) * a_l.mean()
    return f_history


if __name__ == '__main__':
    g_s = 5; g_n = 30; w = 1.0; run_time = 1000; init_time = 100
    c = 1.0; mu = 0.01
    # gamma = 0.5; r = gamma * g_s
    ind_pos, pos_ind = build_structure(g_s, g_n)
    gamma_l = np.round(np.arange(0.1, 1.51, 0.05), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for gamma in gamma_l:
        print(gamma)
        f_0 = [0.5 for _ in range(g_n)]
        history_sim_r = run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w, mu)
        gamma_frac_history.extend(history_sim_r)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
    gamma_frac_history_pd.to_csv('./results/pgg_original_well_mixed.csv')
    print(gamma_frac_history_pd)

