import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import matplotlib.pyplot as plt


def calc_payoff(n, c, r):
    """
    calculate the payoff matrix
    :param n: number of individuals in one group
    :param c: contribution of cooperator
    :param r: enhancement factor
    :return:
    payoff matrix
    """
    payoff = np.zeros((n + 1, 2))  # There are n+1 states (1, 2, 3, ... , n+1 states)
    for i in range(n + 1):
        if i == 0:
            payoff[i][0] = 0
            payoff[i][1] = 0
        else:
            payoff[i][0] = i * r * n * c / n  # payoff of defectors
            payoff[i][1] = i * r * n * c / n - c # payoff of cooperators
        # payoff[i][0] = i * r * n * c / n  # payoff of defectors
        # payoff[i][1] = i * r * n * c / n - c  # payoff of cooperators
    return payoff


def initialize_c_dist(m, n, init_type = 'homo'):
    c_dist = np.zeros((m, n + 1))
    if init_type == 'homo':
        ind_c_p = [0.5 for _ in range(m)]
    elif init_type == 'hete':
        ind_c_p = [_ / m for _ in range(m)]
    for pos in range(m):
        for i in range(n + 1):
            c_dist[pos][i] = binom.pmf(i, n, ind_c_p[pos])
    return c_dist


def t_plus(pos_i, c_num, m, n, c_dist, payoff, mu):
    t_plus_p = 0
    for pos_j in range(m):
        if pos_i != pos_j:
            t_plus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * (1 / (m - 1)) \
                              * (1 / (1 + math.e ** (2.0 * (payoff[c_num][0] - payoff[c_j][1]))))
            t_plus_p += t_plus_p_j
    t_plus_p = (1 - mu) * t_plus_p + mu * (n - c_num) / n
    return t_plus_p


def t_minus(pos_i, c_num, m, n, c_dist, payoff, mu):
    t_minus_p = 0
    for pos_j in range(m):
        if pos_i != pos_j:
            t_minus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * (1 / (m - 1)) \
                               * (1 / (1 + math.e ** (2.0 * (payoff[c_num][1] - payoff[c_j][0]))))
            t_minus_p += t_minus_p_j
    t_minus_p = (1 - mu) * t_minus_p + mu * c_num / n
    return t_minus_p


def calc_trans_matrix(m, n, c_dist, payoff, mu):
    w = np.zeros((m, n + 1, n + 1))
    for pos_i in range(m):
        for c_i in range(n + 1):
            t_plus_p = t_plus(pos_i, c_i, m, n, c_dist, payoff, mu)
            t_minus_p = t_minus(pos_i, c_i, m, n, c_dist, payoff, mu)
            if c_i == 0:
                w[pos_i, c_i, c_i] = 1 - t_plus_p
                w[pos_i, c_i, c_i + 1] = t_plus_p
            elif c_i == n:
                w[pos_i, c_i, c_i - 1] = t_minus_p
                w[pos_i, c_i, c_i] = 1 - t_minus_p
            else:
                w[pos_i, c_i, c_i - 1] = t_minus_p
                w[pos_i, c_i, c_i] = 1 - t_minus_p - t_plus_p
                w[pos_i, c_i, c_i + 1] = t_plus_p
    return w


def dynamic_one_round(m, c_dist, w):
    new_c_dist = []
    for pos in range(m):
        new_c_dist.append(np.dot(c_dist[pos], w[pos]))
    new_c_dist = np.array(new_c_dist)
    return np.array(new_c_dist)


def calc_group_frac(m, n, c_dist):
    group_c_frac = []
    for pos in range(m):
        group_c = 0
        for i in range(n + 1):
            group_c += c_dist[pos][i] * i
        group_c_frac.append(group_c / n)
    return np.array(group_c_frac)


def dynamic_process(m, n, c, r, mu, run_t, init_type):
    payoff = calc_payoff(n, c, r)
    c_dist = initialize_c_dist(m, n, init_type)
    w = calc_trans_matrix(m, n, c_dist, payoff, mu)
    group_c_frac_history = []
    for step in range(run_t):
        group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff, mu)
    group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    return group_c_frac_history


if __name__ == '__main__':
    g_n = 32; g_s = 8; c = 1.0; mu = 0.01; run_time = 1000
    init_type = 'hete'
    gamma_l = np.round(np.arange(0.1, 1.51, 0.05), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for r in gamma_l:
        print(r)
        group_c_frac_history = dynamic_process(g_n, g_s, c, r, mu, run_time, init_type)
        gamma_frac_history.extend(group_c_frac_history)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
    csv_file_name = './results/pgg_original_dynamics_%s.csv' % init_type
    gamma_frac_history_pd.to_csv(csv_file_name)
    print(gamma_frac_history_pd)


