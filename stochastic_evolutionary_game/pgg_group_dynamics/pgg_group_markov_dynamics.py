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
        payoff[i][0] = i * r * n * c / n  # payoff of defectors
        payoff[i][1] = i * r * n * c / n - c # payoff of cooperators
    return payoff


def initialize_c_dist(m, n):
    c_dist = np.zeros((m, n + 1))
    ind_c_p = [0.5 for _ in range(m)]
    for pos in range(m):
        for i in range(n+1):
            c_dist[pos][i] = binom.pmf(i, n, ind_c_p[pos])
    return c_dist


def t_plus(c_num, m, n, c_dist, payoff):
    t_plus_p = 0
    for pos in range(m):
        t_plus_p_j = 0
        c_group_dist = c_dist[pos]
        for c_j in range(n + 1):
            t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * (1 / m) \
                          * (1 / (1 + math.e ** (2.0 * (payoff[c_num][0] - payoff[c_j][1]))))
        t_plus_p += t_plus_p_j
    return t_plus_p


def t_minus(c_num, m, n, c_dist, payoff):
    t_minus_p = 0
    for pos in range(m):
        t_minus_p_j = 0
        c_group_dist = c_dist[pos]
        for c_j in range(n + 1):
            t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * (1 / m) \
                           * (1 / (1 + math.e ** (2.0 * (payoff[c_num][1] - payoff[c_j][0]))))
        t_minus_p += t_minus_p_j
    return t_minus_p


def calc_trans_matrix(m, n, c_dist, payoff):
    w = np.zeros((m, n + 1, n + 1))
    for pos in range(m):
        for c_i in range(n + 1):
            t_plus_p = t_plus(c_i, m, n, c_dist, payoff)
            t_minus_p = t_minus(c_i, m, n, c_dist, payoff)
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
    return np.array(new_c_dist)


def dynamic_process(m, n, c, r, run_t):
    payoff = calc_payoff(n, c, r)
    c_dist = initialize_c_dist(m, n)
    w = calc_trans_matrix(m, n, c_dist, payoff)
    for step in range(run_t):
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff)
    return c_dist


if __name__ == '__main__':
    m = 8; n = 16; c = 1.0; r = 1.2; run_time = 2000
    c_dist_result = dynamic_process(m, n, c, r, run_time)
    c_dist_result_pd = pd
    print(c_dist_result)
    # payoff = calc_payoff(n, c, r)
    # c_dist = initialize_c_dist(m, n)
    # w = calc_trans_matrix(m, n, c_dist, payoff)
    # print(c_dist[0])
    # print(w[0])
    # r = np.dot(c_dist[0], w[0])

    # print(c_dist[0])
    # print(r)
    # r = np.dot(r, w[0])
    # print(r)
    # print(payoff)
    # print(c_dist)


