import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import networkx as nx

def pgg_game(a_l, gamma):  # r in (0, 1]
    a_n = len(a_l)
    p = np.array([np.sum(a_l) * gamma * a_n / a_n for _ in range(a_n)]) - np.array(a_l)
    return p


def generate_lattice(xdim, ydim):
    g_network = nx.grid_2d_graph(xdim, ydim, periodic=True)
    adj_array = nx.to_numpy_array(g_network)
    adj_link = []
    for i in range(adj_array.shape[0]):
        adj_link.append(np.where(adj_array[i] == 1)[0])
    g_edge = nx.Graph()
    for i in range(len(adj_link)):
        for j in range(len(adj_link[i])):
            g_edge.add_edge(i, adj_link[i][j])
    return np.array(adj_link), np.array(g_edge.edges())


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


def game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w, mu, adj_link):
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
    for pos in range(pos_n):
        if np.random.random() < mu:
            g_ind = pos_ind[pos]
            ind = np.random.choice(g_ind)
            ind_a_l[ind] = 1 - ind_a_l_old[ind]
        else:
            g_ind = pos_ind[pos]
            ind = np.random.choice(g_ind)
            potential_pos = adj_link[pos]
            oppon_pos = np.random.choice(potential_pos)
            oppon_ind = pos_ind[oppon_pos]
            while True:
                oppon = np.random.choice(oppon_ind)
                if oppon != ind:
                    break
            ind_p = ind_p_l[ind]
            oppon_p = ind_p_l[oppon]
            t1 = 1 / (1 + math.e ** (2.0 * (ind_p - oppon_p)))
        # t1 = (1 / 2 + w / (2 * delta_pi) * (oppon_p - ind_p))
            t2 = random.random()
            if t2 < t1:
                ind_a_l[ind] = ind_a_l_old[oppon]
    return ind_a_l.reshape((pos_n, int(ind_n / pos_n)))


def run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w, mu, adj_link):
    f_history = []
    for round in range(init_time):
        a_l = initial_action(f_0, ind_pos, pos_ind)
        for step in range(run_time):
            if round == 0:
                f_history.append(a_l.mean(axis=1))
            else:
                f_history[step] = round / (round + 1) * f_history[step] + 1 / (round + 1) * a_l.mean(axis=1)
            a_l = game_one_round(a_l, gamma, ind_pos, pos_ind, g_s, w, mu, adj_link)
        if round == 0:
            f_history.append(a_l.mean(axis=1))
        else:
            f_history[run_time] = round / (round + 1) * f_history[run_time] + 1 / (round + 1) * a_l.mean(axis=1)
    return f_history


if __name__ == '__main__':
    adj, edge = generate_lattice(8, 4)
    g_s = 8; g_b = 2; g_l = 6; w = 1.0; run_time = 1000; init_time = 100
    g_n = g_b ** (g_l - 1); c = 1.0; mu = 0.01
    # gamma = 0.5; r = gamma * g_s
    ind_pos, pos_ind = build_structure(g_s, g_b, g_l)
    gamma_l = np.round(np.arange(0.1, 1.6, 0.1), 2)
    step_l = np.arange(run_time + 1)
    gamma_frac_history = []
    for gamma in gamma_l:
        print(gamma)
        f_0 = [0.5 for _ in range(g_n)]
        history_sim_r = run_game(f_0, init_time, run_time, gamma, ind_pos, pos_ind, g_s, w, mu, adj)
        gamma_frac_history.extend(history_sim_r)
    m_index = pd.MultiIndex.from_product([gamma_l, step_l], names=['gamma', 'step'])
    gamma_frac_history_pd = pd.DataFrame(gamma_frac_history, index=m_index)
    gamma_frac_history_pd.to_csv('./results/pgg_original_lattice.csv')
    print(gamma_frac_history_pd)

