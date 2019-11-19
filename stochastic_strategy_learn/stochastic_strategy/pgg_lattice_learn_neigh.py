import numpy as np
import pandas as pd
import networkx as nx
import math
import random
import datetime


def pgg_game(c_l, gamma):
    a_n = len(c_l)
    p_l = np.array([np.sum(c_l) * gamma * a_n / a_n for _ in range(a_n)]) - np.array(c_l)
    return p_l


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


class Agent:
    def __init__(self, agent_id, link, strategy, contribution):
        self.agent_id = agent_id
        self.link = link
        self.strategy = strategy
        self.ostrategy = strategy
        self.contribution = contribution
        self.payoffs = 0

    def get_id(self):
        return self.agent_id

    def get_link(self):
        return self.link[:]

    def get_strategy(self):
        return self.strategy

    def get_ostrategy(self):
        return self.ostrategy

    def get_contribution(self):
        return self.contribution

    def get_payoffs(self):
        return self.payoffs

    def set_strategy(self, other_strategy):
        self.strategy = other_strategy

    def set_ostrategy(self):
        self.ostrategy = self.strategy

    def set_contribution(self, new_contribution):
        self.contribution = new_contribution

    def set_payoffs(self, p):
        self.payoffs = p

    def add_payoffs(self, p):
        self.payoffs = self.payoffs + p


def initialize_population(popu_size, adj_link, edge, contribution):  # lattice population
    popu = []
    for i in range(popu_size):
        popu.append(Agent(i, adj_link[i], 0.5, contribution))
    return popu


def evolution_one_step(popu, gamma):
    total_num = len(popu)
    for i in range(total_num):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(total_num)]  # action list
    c_l = [0 for _ in range(total_num)]  # contribution_list
    for i in range(total_num):
        if np.random.random() < popu[i].get_strategy():  # strategy is the probability to cooperate
            a_l[i] = 1
            c_l[i] = popu[i].get_contribution()
        else:
            a_l[i] = 0
            c_l[i] = 0
    for i in range(total_num):
        neigh = popu[i].get_link()
        neigh = np.append(neigh, i)  # public goods game, add the focal individual himself to the group
        a_neigh = []
        c_neigh = []
        for j in neigh:
            a_neigh.append(a_l[j])
            c_neigh.append(c_l[j])
        p_neigh_l = pgg_game(c_neigh, gamma)
        for j in range(len(neigh)):
            popu[neigh[j]].add_payoffs(p_neigh_l[j])
    # Update the strategy by imitating other's strategy
    for i in range(total_num):
        ind = popu[i]
        a_i = a_l[i]
        s_i = ind.get_strategy()
        p_i = ind.get_payoffs()
        neigh_i = ind.get_link()
        neigh_i_num = len(neigh_i)
        s_g_i = 0
        for j in neigh_i:
            oppon = popu[j]
            a_j = a_l[j]
            p_j = oppon.get_payoffs()
            s_g_i += s_i * (1 - s_i) * (a_i - a_j) * (0.5 - (1 / (1 + math.e ** (2 * (p_i - p_j))))) / neigh_i_num
        s_i = s_i + s_g_i
        ind.set_strategy(s_i)
    return popu


def run(popu_size, contribution, adj_link, edge, run_time, sample_time, gamma):
    popu = initialize_population(popu_size, adj_link, edge, contribution)
    total_num = len(popu)
    for _ in range(run_time):
        popu = evolution_one_step(popu, gamma)
    sample_strategy = []
    for _ in range(sample_time):
        popu = evolution_one_step(popu, gamma)
        c_frac = 0
        for i in range(total_num):
            c_frac += popu[i].get_strategy()
        c_frac = c_frac / total_num
        sample_strategy.append(c_frac)
    return np.mean(sample_strategy, axis=0)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    popu_size = 100
    c = 1.0
    run_time = 1000
    sample_time = 20
    init_num = 5
    gamma_l = np.round(np.arange(0.1, 2.01, 0.1), 2)
    result_gamma = []
    for gamma_r in gamma_l:
        print(gamma_r)
        result = []
        for _ in range(init_num):
            adj, edge = generate_lattice(10, 10)
            sample_result = run(popu_size, c, adj, edge, run_time, sample_time, gamma_r)
            result.append(sample_result)
        result_gamma.append(np.mean(result, axis=0))
    result_gamma_pd = pd.DataFrame(result_gamma, index=gamma_l, columns=['c_frac'])
    file_name = './results/pgg_lattice_learn_neigh.csv'
    result_gamma_pd.to_csv(file_name)
    end_time = datetime.datetime.now()
    print(end_time - start_time)