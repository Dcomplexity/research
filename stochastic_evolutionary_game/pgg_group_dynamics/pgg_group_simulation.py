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
        group_ind_a = np.random.choice([0, 1], group_ind_n, p=[1 - f_0[pos], f_0[pos]])
        init_a.append(group_ind_a)
    return np.array(init_a)


def game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w):
    delta_pi = ((g_s - 1) * 1.0 * gamma * g_s) / g_s - ((1 * 1.0 * gamma * g_s) / g_s - 1.0)
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
    for pos in range(pos_n):
        ind_p_l = p_l.flatten()
        g_ind = pos_ind[pos]
        ind = np.random.choice(g_ind)
        while True:
            oppon = random.choice(range(ind_n))
            if oppon != ind:
                break
        ind_p = ind_p_l[ind]
        oppon_p = ind_p_l[oppon]
        t1 = (1 / 2 + w / (2 * delta_pi) * (oppon_p - ind_p))
        t2 = random.random()
        if t2 < t1:
            ind_a_l[ind] = ind_a_l_old[oppon]
    return ind_a_l.reshape((pos_n, int(ind_n/pos_n)))


def run_game(f_0, run_time, gamma, ind_pos, pos_ind, g_s, w):
    init_time = 100
    f_history = []
    for round in range(init_time):
        print(round)
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


def T_plus (f, m, n, w, c, r):
    """
    Calculate the probability of increasing one cooperator.
    :param f: the fraction of cooperators in each group
    :param m: the number of group
    :param n: the size of one group
    :param w: the intensity of selection
    :return:
    t_plus_list: a list, the probability that the number of cooperators in each group increase one.
    """
    delta_pi = ((n-1) * c * r) / n - ((1 * c * r) / n - c)
    t_plus_list = np.zeros(m)
    for i in range(m):
        t_plus_value = 0
        for j in range(m):
            t_plus_value += (1 - f[i]) * f[j] / m * (1 / 2 + w / (2 * delta_pi) * (f[j] * r - f[i] * r - c))
        t_plus_list[i] = t_plus_value
    return t_plus_list


def T_minus(f, m, n, w, c, r):
    """
    Calculate the probability of increasing one cooperator.
    :param f: the fraction of cooperators in each group
    :param m: the number of group
    :param n: the size of one group
    :param w: the intensity of selection
    :return:
    """
    delta_pi = ((n-1) * c * r) / n - ((1 * c * r) / n - c)
    t_minus_list = np.zeros(m)
    for i in range(m):
        t_minus_value = 0
        for j in range(m):
            t_minus_value += f[i] * (1 - f[j]) / m * (1 / 2 + w / (2 * delta_pi) * (f[j] * r - f[i] * r + c))
        t_minus_list[i] = t_minus_value
    return t_minus_list


def dynamic_process(f_0, m, n, w, c, r, run_t):
    f_history = []
    f = f_0
    for t in range(run_t):
        f_history.append(np.copy(f))
        t_plus = T_plus(f, m, n, w, c, r)
        t_minus = T_minus(f, m, n, w, c, r)
        f = f + (t_plus - t_minus) * (1 / n)
        for i in range(m):
            if f[i] < 0.0:
                f[i] = 0.0
            elif f[i] > 1.0:
                f[i] = 1.0
    f_history.append(np.copy(f))
    return f_history

if __name__ == '__main__':
    g_s = 20; g_b = 2; g_l = 4; w = 1.0; run_time = 1000; gamma = 0.5
    g_n = g_b ** (g_l - 1); c = 1.0; r = gamma * g_s
    ind_pos, pos_ind = build_structure(g_s, g_b, g_l)
    f_0 = np.random.random(g_n)
    # init_a = initial_action(ind_pos, pos_ind)
    history_sim_r = run_game(f_0, run_time, gamma, ind_pos, pos_ind, g_s, w)
    history_sim_pd = pd.DataFrame(history_sim_r)
    history_sim_pd.plot()
    history_dy_r = dynamic_process(f_0, g_n, g_s, w, c, r, run_time)
    history_dy_pd = pd.DataFrame(history_dy_r)
    history_dy_pd.plot()
    plt.show()
