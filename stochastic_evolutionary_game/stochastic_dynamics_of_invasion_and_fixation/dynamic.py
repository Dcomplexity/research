import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd


def calc_t_plus_prob(c_n, n, r, s, t, p, beta):
    pi_c = (c_n - 1) * r + (n - c_n) * s
    pi_d = c_n * t + (n - c_n - 1) * p
    t_plus_prob = (c_n / n) * ((n - c_n) / n) * (1 / (1 + math.e ** (-1 * beta * (pi_c - pi_d))))
    return t_plus_prob


def calc_t_minus_prob(c_n, n, r, s, t, p, beta):
    pi_c = (c_n - 1) * r + (n - c_n) * s
    pi_d = c_n * t + (n - c_n - 1) * p
    t_minus_prob = (c_n / n) * ((n - c_n) / n) * (1 / (1 + math.e ** (1 * beta * (pi_c - pi_d))))
    return t_minus_prob


def dynamic(init_c_n, n, r, s, t, p, beta, run_time):
    c_n = init_c_n
    for i in range(run_time):
        print(calc_t_plus_prob(c_n, n, r, s, t, p, beta))
        print(calc_t_minus_prob(c_n, n, r, s, t, p, beta))
        c_n = calc_t_plus_prob(c_n, n, r, s, t, p, beta) * (c_n + 1) + calc_t_minus_prob(c_n, n, r, s, t, p, beta) * (c_n - 1)
        + (1 - calc_t_plus_prob(c_n, n, r, s, t, p, beta) - calc_t_minus_prob(c_n, n, r, s, t, p, beta)) * c_n
    return c_n


if __name__ == '__main__':
    init_c_n_r = 19
    n_r = 20
    r_r = 3; s_r = 1; t_r = 4; p_r = 2
    beta_r = 0.01
    run_time_r = 100
    c_n_r = dynamic(init_c_n_r, n_r, r_r, s_r, t_r, p_r, beta_r, run_time_r)
    print(c_n_r)