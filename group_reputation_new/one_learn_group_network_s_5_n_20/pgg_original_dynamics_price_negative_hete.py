import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def price_model(n, m, r):
    p = 1 / (r - 1)
    G = nx.Graph()
    t0 = 3
    all_node = np.arange(n)
    node_array = []
    for i in range(t0):
        for j in range(t0):
            if i != j:
                G.add_edge(i, j)
                node_array.append(j)
    for t in range(t0, n):
        to_link_list = []
        m_flag = 0
        while m_flag < m:
            if random.random() < p:
                to_link_node = np.random.choice(node_array)
                if to_link_node not in to_link_list and t != to_link_node:
                    if (to_link_node, t) not in G.edges and (t, to_link_node) not in G.edges:
                        G.add_edge(t, to_link_node)
                        to_link_list.append(to_link_node)
                        m_flag += 1
            else:
                to_link_node = np.random.choice(all_node)
                if to_link_node not in to_link_list and t != to_link_node:
                    if (to_link_node, t) not in G.edges and (t, to_link_node) not in G.edges:
                        G.add_edge(t, to_link_node)
                        to_link_list.append(to_link_node)
                        m_flag += 1
        node_array.extend(to_link_list)
    return G.to_undirected()


def generate_price(n, m, r):
    g_network = price_model(n, m, r)
    adj_array = nx.to_numpy_array(g_network)
    adj_link = []
    for i in range(adj_array.shape[0]):
        adj_link.append(np.where(adj_array[i] == 1)[0])
    g_edge = nx.Graph()
    for i in range(len(adj_link)):
        for j in range(len(adj_link[i])):
            g_edge.add_edge(i, adj_link[i][j])
    return np.array(adj_link), np.array(g_edge.edges())


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
        elif i == n:
            payoff[i][0] = 0
            payoff[i][1] = i * r * n * c / n - c
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
    elif init_type == 'positive_hete':
        ind_c_p = [(m - 1 - _ + 0.001) / m for _ in range(m)]
    elif init_type == 'negative_hete':
        ind_c_p = [(_ + 0.001) / m for _ in range(m)]
    for pos in range(m):
        for i in range(n + 1):
            c_dist[pos][i] = binom.pmf(i, n, ind_c_p[pos])
    return c_dist


def t_plus(pos_i, c_num, m, n, c_dist, payoff, mu, adj_matrix):
    t_plus_p = 0
    neigh = adj_matrix[pos_i][:]
    neigh_num = np.sum(neigh) + 1
    for pos_j in range(m):
        if pos_i != pos_j:
            t_plus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * neigh[pos_j] * (1 / neigh_num) \
                              * (1 / (1 + math.e ** (2.0 * (payoff[c_num][0] - payoff[c_j][1]))))
            t_plus_p += t_plus_p_j
        else:
            t_plus_p_i = ((n - c_num) / n) * (c_num / n) * (1 / neigh_num) \
                         * (1 / (1 + math.e ** (2.0 * (payoff[c_num][0] - payoff[c_num][1]))))
            t_plus_p += t_plus_p_i
    t_plus_p = (1 - mu) * t_plus_p + mu * (n - c_num) / n
    return t_plus_p


def t_minus(pos_i, c_num, m, n, c_dist, payoff, mu, adj_matrix):
    t_minus_p = 0
    neigh = adj_matrix[pos_i][:]
    neigh_num = np.sum(neigh) + 1
    for pos_j in range(m):
        if pos_i != pos_j:
            t_minus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * neigh[pos_j] * (1 / neigh_num) \
                               * (1 / (1 + math.e ** (2.0 * (payoff[c_num][1] - payoff[c_j][0]))))
            t_minus_p += t_minus_p_j
        else:
            t_minus_p_i = (c_num / n) * ((n - c_num) / n) * (1 / neigh_num) \
                          * (1 / (1 + math.e ** (2.0 * (payoff[c_num][1] - payoff[c_num][0]))))
            t_minus_p += t_minus_p_i
    t_minus_p = (1 - mu) * t_minus_p + mu * c_num / n
    return t_minus_p


def calc_trans_matrix(m, n, c_dist, payoff, mu, adj_matrix):
    w = np.zeros((m, n + 1, n + 1))
    for pos_i in range(m):
        for c_i in range(n + 1):
            t_plus_p = t_plus(pos_i, c_i, m, n, c_dist, payoff, mu, adj_matrix)
            t_minus_p = t_minus(pos_i, c_i, m, n, c_dist, payoff, mu, adj_matrix)
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


def dynamic_process(m, n, c, r, mu, run_t, init_type, adj_matrix):
    payoff = calc_payoff(n, c, r)
    c_dist = initialize_c_dist(m, n, init_type)
    w = calc_trans_matrix(m, n, c_dist, payoff, mu, adj_matrix)
    group_c_frac_history = []
    for step in range(run_t):
        group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff, mu, adj_matrix)
    group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    return group_c_frac_history


if __name__ == '__main__':
    g_n = 20; g_s = 5; c = 1.0; mu = 0.01; run_time = 1000; init_time = 100
    init_type = 'negative_hete'
    gamma_l = np.round(np.arange(0.1, 1.51, 0.05), 2)
    step_l = np.arange(run_time + 1)
    for r_value in [2, 2.2, 2.5, 3, 5]:
        print(r_value)
        adj_matrix = np.zeros((g_n, g_n))
        for i in range(init_time):
            price_graph = price_model(g_n, 2, r_value)
            adj_matrix += nx.adjacency_matrix(price_graph)
        adj_matrix = adj_matrix / init_time
        adj_matrix = np.array(adj_matrix)
        gamma_frac_history = []
        for gamma in gamma_l:
            print(gamma)
            group_c_frac_history = dynamic_process(g_n, g_s, c, gamma, mu, run_time, init_type, adj_matrix)
            gamma_frac_history.extend(group_c_frac_history)
        m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
        gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
        csv_file_name = './results/pgg_original_dynamics_price_%s_%.1f.csv' % (init_type, r_value)
        gamma_frac_history_pd.to_csv(csv_file_name)
        print(gamma_frac_history_pd)


