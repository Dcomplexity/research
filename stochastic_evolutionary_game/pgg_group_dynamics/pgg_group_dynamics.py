import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt


def cal_group_payoff(f, m, n, c, r):
    payoff = np.zeros((m, 2))
    for i in range(m):
        payoff[i][0] = np.round(f[i] * n) * r / n
        payoff[i][1] = np.round(f[i] * n) * r / n - c
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
    delta_pi = np.max(p) - np.min(p)
    t_plus_list = np.zeros(m)
    for i in range(m):
        t_plus_value = 0
        for j in range(m):
            # t_plus_value += (1 - f[i]) * f[j] / m * (1 / 2 + w / (2 * delta_pi) * (f[j] * r - f[i] * r - c))
            t_plus_value += (1 - f[i]) * f[j] / m * (1 / 2 + w / (2 * delta_pi) * (p[j][1] - p[i][0]))
            # t_plus_value += (1 - f[i]) * f[j] / m * (1 / (1 + math.e ** (2.0 * (f[i] * r - f[j] * r + c))))
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
            # t_minus_value += f[i] * (1 - f[j]) / m * (1 / (1 + math.e ** (2.0 * (f[i] * r - c - f[j] * r))))
        t_minus_list[i] = t_minus_value
    return t_minus_list


def evolution_process(f_0, m, n, w, c, r, run_t):
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
    m = 8
    n = 8
    w = 1.0
    c = 1.0
    r = 0.8
    r = r * n
    f_0 = np.random.random(m)
    run_t = 100
    history_r = evolution_process(f_0, m, n, w, c, r, run_t)
    history_pd = pd.DataFrame(history_r)
    print(history_pd)
    history_pd.plot()
    plt.show()
