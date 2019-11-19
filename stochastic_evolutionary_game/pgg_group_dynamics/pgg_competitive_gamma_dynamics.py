import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import matplotlib.pyplot as plt


def initialize_c_dist(m, n):
    c_dist = np.zeros((m, n + 1))
    ind_c_p = [(i + 1) / m for i in range(m)]
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
                    if pos_i != pos_j:
                        for j in range(n + 1):
                            r_temp += r / edge_num * m * (i + 0.001) / (i + j + 0.001 * 2) * c_dist[pos_j][j]
                r_l[pos_i][i] = r_temp
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
    for pos in range(m):
        t_plus_p_j = 0
        c_group_dist = c_dist[pos]
        for c_j in range(n + 1):
            t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * (1 / m) \
                          * (1 / (1 + math.e ** (2.0 * (payoff[pos_i][c_num][0] - payoff[pos][c_j][1]))))
        t_plus_p += t_plus_p_j
    t_plus_p = (1 - mu) * t_plus_p + mu * (n - c_num) / n
    return t_plus_p


def t_minus(pos_i, c_num, m, n, c_dist, payoff, mu):
    t_minus_p = 0
    for pos in range(m):
        t_minus_p_j = 0
        c_group_dist = c_dist[pos]
        for c_j in range(n + 1):
            t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * (1 / m) \
                           * (1 / (1 + math.e ** (2.0 * (payoff[pos_i][c_num][1] - payoff[pos][c_j][0]))))
        t_minus_p += t_minus_p_j
    t_minus_p = (1 - mu) * t_minus_p + mu * c_num / n
    return t_minus_p


def calc_trans_matrix(m, n, c_dist, payoff, mu):
    w = np.zeros((m, n + 1, n + 1))
    for pos in range(m):
        for c_i in range(n + 1):
            t_plus_p = t_plus(pos, c_i, m, n, c_dist, payoff, mu)
            t_minus_p = t_minus(pos, c_i, m, n, c_dist, payoff, mu)
            if c_i == 0:
                w[pos, c_i, c_i] = 1 - t_plus_p
                w[pos, c_i, c_i + 1] = t_plus_p
            elif c_i == n:
                w[pos, c_i, c_i - 1] = t_minus_p
                w[pos, c_i, c_i] = 1 - t_minus_p
            else:
                w[pos, c_i, c_i - 1] = t_minus_p
                w[pos, c_i, c_i] = 1 - t_minus_p - t_plus_p
                w[pos, c_i, c_i + 1] = t_plus_p
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


def dynamic_process(m, n, c, r, mu, run_t):
    c_dist = initialize_c_dist(m, n)
    r_l = calc_enhancement_l(m, n, c_dist, r, average=True)
    payoff = calc_payoff(m, n, c, r_l)
    w = calc_trans_matrix(m, n, c_dist, payoff, mu)
    group_c_frac_history = []
    group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    for step in range(run_t):
        print(step)
        r_l = calc_enhancement_l(m, n, c_dist, r, average=False)
        payoff = calc_payoff(m, n, c, r_l)
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff, mu)
        group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    group_c_frac_history_pd = pd.DataFrame(group_c_frac_history)
    group_c_frac_history_pd.to_csv('./results_old/pgg_competitive_gamma.csv')
    return group_c_frac_history_pd


if __name__ == '__main__':
    m = 64; n = 8; c = 1.0; r = 0.7; mu = 0.01; run_time = 200
    group_c_frac_history = dynamic_process(m, n, c, r, mu, run_time)
    print(group_c_frac_history)


