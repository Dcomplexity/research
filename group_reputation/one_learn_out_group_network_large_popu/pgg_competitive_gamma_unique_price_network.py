import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse


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


def calc_enhancement_l(m, n, c_dist, r, adj_link, edge, average=False):
    r_l = np.zeros((m, n + 1))
    if average == True:
        for pos in range(m):
            for i in range(n + 1):
                r_l[pos][i] = r
    else:
        edge_num = len(edge)
        for pos_i in range(m):
            neigh = adj_link[pos_i]
            for i in range(n + 1):
                r_temp = 0
                for pos_j in neigh:
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


def t_plus(pos_i, c_num, m, n, c_dist, payoff, mu, adj_link):
    t_plus_p = 0
    neigh = adj_link[pos_i]
    neigh_num = len(neigh)
    for pos_j in neigh:
        if pos_i != pos_j:
            t_plus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_plus_p_j += ((n - c_num) / n) * (c_j / n) * c_group_dist[c_j] * (1 / neigh_num) \
                              * (1 / (1 + math.e ** (2.0 * (payoff[pos_i][c_num][0] - payoff[pos_j][c_j][1]))))
            t_plus_p += t_plus_p_j
    t_plus_p = (1 - mu) * t_plus_p + mu * (n - c_num) / n
    return t_plus_p


def t_minus(pos_i, c_num, m, n, c_dist, payoff, mu, adj_link):
    t_minus_p = 0
    neigh = adj_link[pos_i]
    neigh_num = len(neigh)
    for pos_j in neigh:
        if pos_i != pos_j:
            t_minus_p_j = 0
            c_group_dist = c_dist[pos_j]
            for c_j in range(n + 1):
                t_minus_p_j += (c_num / n) * ((n - c_j) / n) * c_group_dist[c_j] * (1 / neigh_num) \
                               * (1 / (1 + math.e ** (2.0 * (payoff[pos_i][c_num][1] - payoff[pos_j][c_j][0]))))
            t_minus_p += t_minus_p_j
    t_minus_p = (1 - mu) * t_minus_p + mu * c_num / n
    return t_minus_p


def calc_trans_matrix(m, n, c_dist, payoff, mu, adj_link):
    w = np.zeros((m, n + 1, n + 1))
    for pos_i in range(m):
        for c_i in range(n + 1):
            t_plus_p = t_plus(pos_i, c_i, m, n, c_dist, payoff, mu, adj_link)
            t_minus_p = t_minus(pos_i, c_i, m, n, c_dist, payoff, mu, adj_link)
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


def dynamic_process(m, n, c, r, mu, run_t, init_type, adj_link, edge):
    c_dist = initialize_c_dist(m, n, init_type)
    r_l = calc_enhancement_l(m, n, c_dist, r, adj_link, edge, average=True)
    payoff = calc_payoff(m, n, c, r_l)
    w = calc_trans_matrix(m, n, c_dist, payoff, mu, adj_link)
    group_c_frac_history = []
    group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    for step in range(run_t):
        r_l = calc_enhancement_l(m, n, c_dist, r, adj_link, edge, average=False)
        payoff = calc_payoff(m, n, c, r_l)
        c_dist = dynamic_one_round(m, c_dist, w)
        w = calc_trans_matrix(m, n, c_dist, payoff, mu, adj_link)
        group_c_frac_history.append(np.copy(calc_group_frac(m, n, c_dist)))
    return group_c_frac_history


def pgg_game(a_l, gamma):  # r in (0, 1]
    a_n = len(a_l)
    p = np.array([np.sum(a_l) * gamma * a_n / a_n for _ in range(a_n)]) - np.array(a_l)
    return p


class SocialStructure():
    def __init__(self, g_s, g_n, t_n):
        self.g_s = g_s  # group_size
        self.g_n = g_n  # group_num
        self.t_n = t_n  # total_num
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


def build_structure(g_s, g_n):
    t_n = g_s * g_n
    s_s = SocialStructure(g_s, g_n, t_n)
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


def initial_gamma(pos_n, init_gamma_value):
    init_gamma = [init_gamma_value for _ in range(pos_n)]
    return init_gamma


def game_one_round(a_l, g_gamma_l, ind_pos, pos_ind, g_s, w, ave_gamma, mu, adj_link, edge):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = np.copy(a_l)
    ind_a_l = a_l.flatten()
    ind_a_l_old = a_l_old.flatten()
    p_l = []
    g_a_frac = np.zeros(pos_n)
    for pos in range(pos_n):
        g_a = np.copy(a_l[pos])
        g_p = pgg_game(g_a, g_gamma_l[pos])
        p_l.append(g_p)
        g_a_frac[pos] = np.mean(g_a)
    p_l = np.array(p_l)
    ind_p_l = p_l.flatten()
    for pos in range(pos_n):
        if random.random() < mu:
            g_ind = pos_ind[pos]
            ind = random.choice(g_ind)
            ind_a_l[ind] = int(1 - ind_a_l_old[ind])
        else:
            g_ind = pos_ind[pos]
            ind = random.choice(g_ind)
            potential_pos = adj_link[pos]
            oppon_pos = random.choice(potential_pos)
            oppon_ind = pos_ind[oppon_pos]
            while True:
                oppon = random.choice(oppon_ind)
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
    edge_num = len(edge)
    for pos_i in range(pos_n):
        for pos_j in adj_link[pos_i]:
            if pos_i == pos_j:
                print("wrong")
            if pos_i != pos_j:
                g_gamma_l[pos_i] += ave_gamma / edge_num * pos_n * (g_a_frac[pos_i] + 0.001) \
                                   / (g_a_frac[pos_i] + g_a_frac[pos_j] + 0.002)
    return ind_a_l.reshape((pos_n, int(ind_n / pos_n))), g_gamma_l


def run_game(f_0, init_time, run_time, ave_gamma, ind_pos, pos_ind, g_s, w, mu, adj_link, edge):
    f_history = []
    pos_n = len(pos_ind)
    for round in range(init_time):
        a_l = initial_action(f_0, ind_pos, pos_ind)
        g_gamma_l = initial_gamma(pos_n, ave_gamma)
        for step in range(run_time):
            if round == 0:
                f_history.append(a_l.mean(axis=1))
            else:
                f_history[step] = round / (round + 1) * f_history[step] + 1 / (round + 1) * a_l.mean(axis=1)
            a_l, g_gamma_l = game_one_round(a_l, g_gamma_l, ind_pos, pos_ind, g_s, w, ave_gamma, mu, adj_link, edge)
        if round == 0:
            f_history.append(a_l.mean(axis=1))
        else:
            f_history[run_time] = round / (round + 1) * f_history[run_time] + 1 / (round + 1) * a_l.mean(axis=1)
    return f_history


if __name__ == '__main__':
    g_n = 32; g_s = 8; w = 1.0; c = 1.0; mu = 0.01; run_time = 1000; init_time = 100
    init_type = 'homo'
    ind_pos, pos_ind = build_structure(g_s, g_n)
    gamma_l = np.round(np.arange(0.1, 1.51, 0.05), 2)
    step_l = np.arange(run_time + 1)
    for r_value in [2, 2.2, 2.5, 3, 5]:
        print(r_value)
        gamma_frac_history_simulation = []
        gamma_frac_history_dynamics = []
        adj_link, edge = generate_price(g_n, 2, r_value)
        for gamma in gamma_l:
            print(gamma)
            f_0 = [0.5 for _ in range(g_n)]
            group_c_frac_history_sim_r = run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w, mu,
                                                  adj_link, edge)
            group_c_frac_history_dy_r = dynamic_process(g_n, g_s, c, gamma, mu, run_time, init_type, adj_link, edge)
            gamma_frac_history_simulation.extend(group_c_frac_history_sim_r)
            gamma_frac_history_dynamics.extend(group_c_frac_history_dy_r)
        m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
        gamma_frac_history_simulation_pd = pd.DataFrame(gamma_frac_history_simulation, index=m_index)
        gamma_frac_history_dynamics_pd = pd.DataFrame(gamma_frac_history_dynamics, index=m_index)
        csv_file_name_simulation = './results/pgg_competitive_gamma_simulation_unique_price_network_%.1f.csv' % r_value
        gamma_frac_history_simulation_pd.to_csv(csv_file_name_simulation)
        csv_file_name_dynamics = './results/pgg_competitive_gamma_dynamics_unique_price_network_%.1f.csv' % r_value
        gamma_frac_history_dynamics_pd.to_csv(csv_file_name_dynamics)


