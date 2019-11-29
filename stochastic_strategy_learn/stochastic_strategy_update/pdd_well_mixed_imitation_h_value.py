import numpy as np
import pandas as pd
import networkx as nx
import math
import random
import datetime


def pdd_game(a_x, a_y, r, s, t, p):
    if a_x == 1 and a_y == 1:
        p_x = r; p_y = r
    elif a_x == 1 and a_y == 0:
        p_x = s; p_y = t
    elif a_x == 0 and a_y == 1:
        p_x = t; p_y = s
    elif a_x == 0 and a_y == 0:
        p_x = p; p_y = p
    else:
        p_x = None; p_y = None
    return (p_x, p_y)


def generate_well_mixed_network(popu_size):
    g_network = nx.complete_graph(popu_size)
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
        self.payoff = 0
        self.ave_payoff = 0

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
        return self.payoff

    def get_ave_payoff(self):
        return self.ave_payoff

    def set_strategy(self, other_strategy):
        self.strategy = other_strategy

    def set_ostrategy(self):
        self.ostrategy = self.strategy

    def set_contribution(self, new_contribution):
        self.contribution = new_contribution

    def set_payoff(self, p):
        self.payoff = p

    def add_payoff(self, p):
        self.payoff = self.payoff + p

    def update_ave_payoff(self, p):
        self.ave_payoff



def initialize_population(popu_size, adj_link, edge, contribution):
    popu = []
    for i in range(popu_size):
        popu.append(Agent(i, adj_link[i], 0.5, contribution))
    return popu


def evolution_one_step(popu, edge, r, s, t, p):
    total_num = len(popu)
    for i in range(total_num):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(total_num)]
    c_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        if np.random.random() < popu[i].get_strategy():
            a_l[i] = 1
            c_l[i] = popu[i].get_contribution()
        else:
            a_l[i] = 0
            c_l[i] = 0
    for pair in edge:
        ind_x = pair[0]
        ind_y = pair[1]
        p_x, p_y = pdd_game(a_l[ind_x], a_l[ind_y], r, s, t, p)
        popu[ind_x].add_payoffs(p_x)
        popu[ind_y].add_payoffs(p_y)
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


def run(popu_size, contribution, adj_link, edge, run_time, sample_time, r, s, t, p):
    popu = initialize_population(popu_size, adj_link, edge, contribution)
    total_num = len(popu)
    for _ in range(run_time):
        popu = evolution_one_step(popu, edge, r, s, t, p)
    sample_strategy = []
    for _ in range(sample_time):
        popu = evolution_one_step(popu, edge, r, s, t, p)
        c_frac = 0
        for i in range(total_num):
            c_frac += popu[i].get_strategy()
        c_frac = c_frac / total_num
        sample_strategy.append(c_frac)
    return np.mean(sample_strategy, axis=0)


if __name__ == '__main__':
    popu_size = 100
    c = 1.0
    run_time = 1000
    sample_time = 20
    init_num = 5
    r = 3; s = 0; t = 5; p = 1
    result_l = []
    for _ in range(init_num):
        adj, edge = generate_well_mixed_network(popu_size)
        sample_result = run(popu_size, c, adj, edge, run_time, sample_time, r, s, t, p)
        result_l.append(sample_result)
    print(np.mean(result_l))
