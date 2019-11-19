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
        group_ind_a = np.random.choice([0, 1], group_ind_n, p = [1 - f_0[pos], f_0[pos]])
        init_a.append(group_ind_a)
    return np.array(init_a)


def game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = np.copy(a_l)
    ind_a_l = a_l.flatten()
    ind_a_l_old = a_l_old.flatten()
    p_l = []
    for pos in range(pos_n):
        g_a = np.copy(a_l[pos])
        g_p = pgg_game(g_a, gamma)
        p_l.append(g_p)
    p_l = np.array(p_l)
    ind_p_l = p_l.flatten()
    delta_pi = np.max(ind_p_l) - np.min(ind_p_l) + 0.001
    for ind in range(ind_n):
        while True:
            oppon = random.choice(range(ind_n))
            if oppon != ind:
                break
        ind_p = ind_p_l[ind]
        oppon_p = ind_p_l[oppon]
        # t1 = (1 / 2 + w / (2 * delta_pi) * (oppon_p - ind_p))
        t1 = 1 / (1 + math.e ** (2.0 * (ind_p - oppon_p)))
        t2 = random.random()
        if t2 < t1:
            ind_a_l[ind] = ind_a_l_old[oppon]
    return ind_a_l.reshape((pos_n, int(ind_n / pos_n)))


def run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w):
    f_history = []
    for round in range(init_time):
        a_l = initial_action(f_0, ind_pos, pos_ind)
        for step in range(run_time):
            if round == 0:
                f_history.append(a_l.mean(axis=1))
            else:
                f_history[step] = round / (round + 1) * f_history[step] + 1 / (round + 1) * a_l.mean(axis=1)
            a_l = game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w)
        if round == 0:
            f_history.append(a_l.mean(axis=1))
        else:
            f_history[run_time] = round / (round + 1) * f_history[run_time] + 1 / (round + 1) * a_l.mean(axis=1)
    return f_history


def cal_group_payoff(f, m, n, c, r):
    payoff = np.zeros((m, 2))
    for i in range(m):
        payoff[i][0] = f[i] * r
        payoff[i][1] = f[i] * r - c
    return payoff


def T_plus (f, m, n, w, c, r, p):
    """
    Calculate the probability of increasing one cooperator.
    :param f: the fraction of cooperators in each group
    :param m: the number of group
    :param n: the size of one group
    :param w: the intensity of selection
    :return:
    t_plus_list: a list, the probability that the number of cooperators in each group increase one.
    """
    # delta_pi = ((n-1) * c * r) / n - ((1 * c * r) / n - c)
    delta_pi = np.max(p) - np.min(p) + 0.001
    t_plus_list = np.zeros(m)
    for i in range(m):
        t_plus_value = 0
        for j in range(m):
            # t_plus_value += (1 - f[i]) * f[j] / m * (1 / 2 + w / (2 * delta_pi) * (f[j] * r - f[i] * r - c))
            t_plus_value += (1 - f[i]) * f[j] / m * (1 / 2 + w / (2 * delta_pi) * (p[j][1] - p[i][0]))
        t_plus_list[i] = t_plus_value
    return t_plus_list


def T_minus(f, m, n, w, c, r, p):
    """
    Calculate the probability of increasing one cooperator.
    :param f: the fraction of cooperators in each group
    :param m: the number of group
    :param n: the size of one group
    :param w: the intensity of selection
    :return:
    """
    # delta_pi = ((n-1) * c * r) / n - ((1 * c * r) / n - c)
    delta_pi = np.max(p) - np.min(p) + 0.001
    t_minus_list = np.zeros(m)
    for i in range(m):
        t_minus_value = 0
        for j in range(m):
            # t_minus_value += f[i] * (1 - f[j]) / m * (1 / 2 + w / (2 * delta_pi) * (f[j] * r - f[i] * r + c))
            t_minus_value += f[i] * (1 - f[j]) / m * (1 / 2 + w / (2 * delta_pi) * (p[j][0] - p[i][1]))
        t_minus_list[i] = t_minus_value
    return t_minus_list


def dynamic_process(f_0, m, n, w, c, r, run_t):
    f_history = []
    f = f_0
    for t in range(run_t):
        f_history.append(np.copy(f))
        p = cal_group_payoff(f, m, n, c, r)
        t_plus = T_plus(f, m, n, w, c, r, p)
        t_minus = T_minus(f, m, n, w, c, r, p)
        f = f + (t_plus - t_minus) * (1 / n)
        for i in range(m):
            if f[i] < 0.0:
                f[i] = 0.0
            elif f[i] > 1.0:
                f[i] = 1.0
    f_history.append(np.copy(f))
    return f_history

if __name__ == '__main__':
    g_s = 8; g_b = 2; g_l = 4; w = 1.0; run_time = 100; init_time = 1000
    g_n = g_b ** (g_l - 1); c = 1.0
    # gamma = 0.5; r = gamma * g_s
    ind_pos, pos_ind = build_structure(g_s, g_b, g_l)
    gamma_l = np.round(np.arange(0.1, 1.9, 0.1), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for gamma in gamma_l:
        print(gamma)
        f_0 = np.random.random(g_n)
        history_sim_r = run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w)
        gamma_frac_history.extend(history_sim_r)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
    gamma_frac_history_pd.to_csv('./results_old/pgg_original.csv')
    print(gamma_frac_history_pd)

