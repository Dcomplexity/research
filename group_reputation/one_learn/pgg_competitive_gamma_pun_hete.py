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
    def __init__(self, g_s, g_b, g_l, t_n):
        self.g_s = g_s  # group_size
        self.g_b = g_b  # group_bae
        self.g_l = g_l  # group_length
        self.t_n = t_n  # total_num
        self.g_n = self.g_b ** (self.g_l - 1)
        if self.t_n != self.g_s * self.g_n:
            print ("Error: The total num of individuals does not correspond to the social structure")

    def build_social_structure(self):
        ind_pos = [0 for x in range(self.t_n)]
        pos_ind = [[] for x in range(self.g_n)]
        for i in range(self.g_n):
            for j in range(i*self.g_s, (i+1)*self.g_s):
                ind_pos[j] = 1
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(g_s, g_b, g_l):
    t_n = g_s * (g_b ** (g_l - 1))
    s_s = SocialStructure(g_s, g_b, g_l, t_n)
    ind_pos, pos_ind = s_s.build_social_structure()
    return ind_pos, pos_ind


def initial_action(f_0, ind_pos, pos_ind):
    init_a = []
    g_n = len(pos_ind)
    for pos in range(g_n):
        group_ind = pos_ind[pos]
        group_ind_n = len(group_ind)
        group_ind_a = np.random.choice([0, 1, 2], group_ind_n, p = [1 - f_0[pos], f_0[pos]/ 2, f_0[pos] / 2])
        init_a.append(group_ind_a)
    return np.array(init_a)


def initial_gamma(pos_n, init_gamma_value):
    init_gamma = [init_gamma_value for _ in range(pos_n)]
    return init_gamma


def game_one_round(a_l, g_gamma_l, ind_pos, pos_ind, g_s, w, ave_gamma, mu):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = np.copy(a_l)
    ind_a_l = a_l.flatten()
    ind_a_l_old = a_l_old.flatten()
    p_l = []
    g_a_frac = np.zeros(pos_n)
    for pos in range(pos_n):
        g_a = np.copy(a_l[pos])
        pgg_a = []
        stra_count = [0, 0, 0]
        for a_i in g_a:
            if a_i == 0:
                pgg_a.append(0)
                stra_count[a_i] += 1
            elif a_i == 1:
                pgg_a.append(1)
                stra_count[a_i] += 1
            elif a_i == 2:
                pgg_a.append(1)
                stra_count[a_i] += 1
        g_p = pgg_game(pgg_a, g_gamma_l[pos])
        if stra_count[0] > 0:
            for i in range(len(g_a)):
                if g_a[i] == 2:
                    g_p[i] -= 1.0
                if g_a[i] == 0:
                    g_p[i] -= 1.0 * stra_count[2] / stra_count[0]
        p_l.append(np.array(g_p))
        g_a_frac[pos] = np.mean(pgg_a)
    p_l = np.array(p_l)
    ind_p_l = p_l.flatten()
    for pos in range(pos_n):
        if np.random.random() < mu:
            g_ind = pos_ind[pos]
            ind = np.random.choice(g_ind)
            poten_a = [0, 1, 2]
            poten_a.remove(ind_a_l_old[ind])
            ind_a_l[ind] = np.random.choice(poten_a)
        else:
            g_ind = pos_ind[pos]
            ind = np.random.choice(g_ind)
            while True:
                oppon = random.choice(range(ind_n))
                if oppon != ind:
                    break
            ind_p = ind_p_l[ind]
            oppon_p = ind_p_l[oppon]
            t1 = 1 / (1 + math.e ** (2.0 * (ind_p - oppon_p)))
            # t1 = (1 / 2 + w / (2 * delta_pi) * (oppon_p - ind_p))
            t2 = random.random()
            if t2 < t1:
                ind_a_l[ind] = ind_a_l_old[oppon]

    # Update g_gamma_l
    g_gamma_l = np.zeros(pos_n)
    edge_num = pos_n * (pos_n - 1) / 2
    for pos_i in range(pos_n):
        for pos_j in range(pos_n):
            if pos_i != pos_j:
                g_gamma_l[pos_i] += ave_gamma / edge_num * pos_n * (g_a_frac[pos_i] + 0.001) \
                                   / (g_a_frac[pos_i] + g_a_frac[pos_j] + 0.002)
    return ind_a_l.reshape((pos_n, int(ind_n / pos_n))), g_gamma_l


def run_game(f_0, init_time, run_time, ave_gamma, ind_pos, pos_ind, g_s, w, mu):
    f_history = []
    pos_n = len(pos_ind)
    for round in range(init_time):
        a_l = initial_action(f_0, ind_pos, pos_ind)
        new_a_frac = np.zeros((pos_n, 3))
        for pos in range(pos_n):
            for a_i in a_l[pos]:
                new_a_frac[pos][a_i] += 1
            new_a_frac[pos] = new_a_frac[pos] / g_s
        g_gamma_l = initial_gamma(pos_n, ave_gamma)
        for step in range(run_time):
            if round == 0:
                f_history.append(new_a_frac)
            else:
                f_history[step] = round / (round + 1) * f_history[step] + 1 / (round + 1) * new_a_frac
            a_l, g_gamma_l = game_one_round(a_l, g_gamma_l, ind_pos, pos_ind, g_s, w, ave_gamma, mu)
            new_a_frac = np.zeros((pos_n, 3))
            for pos in range(pos_n):
                for a_i in a_l[pos]:
                    new_a_frac[pos][a_i] += 1
                new_a_frac[pos] = new_a_frac[pos] / g_s
        new_a_frac = np.zeros((pos_n, 3))
        for pos in range(pos_n):
            for a_i in a_l[pos]:
                new_a_frac[pos][a_i] += 1
            new_a_frac[pos] = new_a_frac[pos] / g_s
        if round == 0:
            f_history.append(new_a_frac)
        else:
            f_history[run_time] = round / (round + 1) * f_history[run_time] + 1 / (round + 1) * new_a_frac
    return f_history


if __name__ == '__main__':
    g_s = 8; g_b = 2; g_l = 6; w = 1.0; run_time = 5000; init_time = 100
    g_n = g_b ** (g_l - 1); c = 1.0; mu = 0.01
    ind_pos, pos_ind = build_structure(g_s, g_b, g_l)
    gamma_l = np.round(np.arange(0.1, 1.6, 0.1), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for ave_gamma in gamma_l:
        print(ave_gamma)
        # f_0 = np.random.random(g_n)
        f_0 = [_ / g_n for _ in range(g_n)]
        history_sim_r = run_game(f_0, init_time, run_time, ave_gamma, ind_pos, pos_ind, g_s, w, mu)
        gamma_frac_history.extend(history_sim_r)
    new_gamma_frac_history = []
    for item in gamma_frac_history:
        new_gamma_frac_history.append(item.flatten())
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(new_gamma_frac_history, index=m_index)
    gamma_frac_history_pd.to_csv('./results_old/pgg_competitive_gamma_pun_hete.csv')
    print(gamma_frac_history_pd)
