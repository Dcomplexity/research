import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def initialize_c_dist(m, n, init_type = 'homo'):
    c_dist = np.zeros((m, n + 1))
    if init_type == 'homo':
        ind_c_p = [0.5 for _ in range(m)]
    elif init_type == 'hete':
        ind_c_p = [(_ + 0.001) / m for _ in range(m)]
    for pos in range(m):
        for i in range(n + 1):
            c_dist[pos][i] = binom.pmf(i, n, ind_c_p[pos])
    return c_dist


def calc_enhancement_l(m, n, c_dist, r, average=False):
    r_l = np.zeros((m, n + 1))
    if average == True:
        for pos in range(m):
            for i in range(n + 1):
                r_l[pos][i] = r
    else:
        edge_num = m * (m - 1) / 2
        for pos_i in range(m):
            for i in range(n + 1):
                r_temp = 0
                for pos_j in range(m):
                    mean_j = 0
                    if pos_i != pos_j:
                        for j in range(n + 1):
                            #mean_j += j * c_dist[pos_j][j]
                            r_temp += r / edge_num * m * (i/n + 0.001) / (i/n + j/n + 0.001 * 2) * c_dist[pos_j][j]
                    #r_temp += r / edge_num * m * (i + 0.001) / (i + mean_j + 0.001 * 2)
                r_l[pos_i][i] = r_temp
    return r_l


def calc_enhancement_w_l(m, n, c_dist, r, w):
    r_l = np.zeros((m, n + 1))
    edge_num = m * (m - 1) / 2
    for pos_i in range(m):
        w_i = w[pos_i]
        c_dist_i = c_dist[pos_i]
        for i in range(n + 1):
            if i == 0:
                s_p = w_i[i][i] * c_dist_i[i] + w_i[i+1][i] * c_dist_i[i+1]
                c_p = np.zeros(n + 1)  # conditional probability
                c_p[i] = w_i[i][i] * c_dist_i[i] / s_p
                c_p[i+1] = w_i[i+1][i] * c_dist_i[i+1] / s_p
            elif i == n:
                s_p = w_i[i][i] * c_dist_i[i] + w_i[i-1][i] * c_dist_i[i-1]
                c_p = np.zeros(n + 1)
                c_p[i] = w_i[i][i] * c_dist_i[i] / s_p
                c_p[i-1] = w_i[i-1][i] * c_dist_i[i-1] / s_p
            else:
                s_p = w_i[i][i] * c_dist_i[i] + w_i[i-1][i] * c_dist_i[i-1] + w_i[i+1][i] * c_dist_i[i+1]
                c_p = np.zeros(n + 1)
                c_p[i] = w_i[i][i] * c_dist_i[i] / s_p
                c_p[i+1] = w_i[i+1][i] * c_dist_i[i+1] / s_p
                c_p[i-1] = w_i[i-1][i] * c_dist_i[i-1] / s_p
            if i == 0:
                for i_last in [i, i+1]:
                    r_temp = 0
                    for pos_j in range(m):
                        if pos_i != pos_j:
                            for j in range(n + 1):
                                r_temp += r * m / edge_num * (i_last / n + 0.001) / (i_last / n + j / n + 0.001 * 2) \
                                          * c_dist[pos_j][j]
                    r_l[pos_i][i] += r_temp * c_p[i_last]
            elif i == n:
                for i_last in [i-1, i]:
                    r_temp = 0
                    for pos_j in range(m):
                        if pos_i != pos_j:
                            for j in range(n + 1):
                                r_temp += r * m / edge_num * (i_last / n + 0.001) / (i_last / n + j / n + 0.001 * 2) \
                                          * c_dist[pos_j][j]
                    r_l[pos_i][i] += r_temp * c_p[i_last]
            else:
                for i_last in [i-1, i, i+1]:
                    r_temp = 0
                    for pos_j in range(m):
                        if pos_i != pos_j:
                            for j in range(n + 1):
                                r_temp += r * m / edge_num * (i_last / n + 0.001) / (i_last / n + j / n + 0.001 * 2) \
                                          * c_dist[pos_j][j]
                    r_l[pos_i][i] += r_temp * c_p[i_last]
    return r_l


def calc_payoff(m, n, c, r_l):
    """
    calculate the payoff matrix
    :param m: number of groups
    :param n: number of individuals in one group
    :param c: contribution of cooperator
    :param r_l: list of enhancement factor belonging to each group
    :return:
    payoff matrix
    """
    payoff = np.zeros((m, n + 1, 2))  # There are n+1 states (1, 2, 3, ... , n+1 states)
    for pos in range(m):
        for i in range(n + 1):
            if i == 0:
                payoff[pos][i][0] = 0
                payoff[pos][i][1] = 0
            elif i == n:
                payoff[pos][i][0] = 0
                payoff[pos][i][1] = i * r_l[pos][i] * n * c / n - c
            else:
                payoff[pos][i][0] = i * r_l[pos][i] * n * c / n  # payoff of defectors
                payoff[pos][i][1] = i * r_l[pos][i] * n * c / n - c # payoff of cooperators
    return payoff


def t_plus(pos_i, c_num, m, n, c_dist, payoff, mu):
    t_plus_p = 0
    for pos_j in range(m):
        if pos_i != pos_j:
            t_plus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * (1 / m) \
                              * (1 / (1 + math.e ** (1.0 * (payoff[pos_i][c_num][0] - payoff[pos_j][c_j][1]))))
            t_plus_p += t_plus_p_j
        else:
            t_plus_p_i = ((n - c_num) / n) * (c_num / n) * (1 / m) \
                         * (1 / (1 + math.e ** (1.0 * (payoff[pos_i][c_num][0] - payoff[pos_i][c_num][1]))))
            t_plus_p += t_plus_p_i
    t_plus_p = (1 - mu) * t_plus_p + mu * (n - c_num) / n
    return t_plus_p


def t_minus(pos_i, c_num, m, n, c_dist, payoff, mu):
    t_minus_p = 0
    for pos_j in range(m):
        if pos_i != pos_j:
            t_minus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * (1 / m) \
                               * (1 / (1 + math.e ** (1.0 * (payoff[pos_i][c_num][1] - payoff[pos_j][c_j][0]))))
            t_minus_p += t_minus_p_j
        else:
            t_minus_p_i = (c_num / n) * ((n - c_num) / n) * (1 / m) \
                          * (1 / (1 + math.e ** (1.0 * (payoff[pos_i][c_num][1] - payoff[pos_i][c_num][0]))))
            t_minus_p += t_minus_p_i
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
    return new_c_dist


def calc_group_frac(m, n, c_dist):
    group_c_frac = []
    for pos in range(m):
        group_c = 0
        for i in range(n + 1):
            group_c += c_dist[pos][i] * i
        group_c_frac.append(group_c / n)
    return np.array(group_c_frac)


def dynamic_process(m, n, c, r, mu, run_t, init_type):
    c_dist = initialize_c_dist(m, n, init_type)
    r_l = calc_enhancement_l(m, n, c_dist, r, average=True)
    payoff = calc_payoff(m, n, c, r_l)
    w = calc_trans_matrix(m, n, c_dist, payoff, mu)
    group_c_frac_history = []
    group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    for step in range(run_t):
        r_l = calc_enhancement_w_l(m, n, c_dist, r, w)
        payoff = calc_payoff(m, n, c, r_l)
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff, mu)
        group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    return group_c_frac_history


if __name__ == '__main__':
    g_n = 30; g_s = 5; c = 1.0; mu = 0.01; run_time = 1000
    init_type = 'homo'
    gamma_l = np.round(np.arange(0.1, 1.51, 0.05), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for r in gamma_l:
        print(r)
        group_c_frac_history = dynamic_process(g_n, g_s, c, r, mu, run_time, init_type)
        gamma_frac_history.extend(group_c_frac_history)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
    csv_file_name = './results_beta_1/pgg_competitive_gamma_dynamics_%s.csv' % init_type
    gamma_frac_history_pd.to_csv(csv_file_name)
    print(gamma_frac_history_pd)


