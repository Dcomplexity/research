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


def gen_actions():
    defect = 0
    cooperate = 1
    actions = [defect, cooperate]
    return actions


class Agent:
    def __init__(self, agent_id, link, actions, contribution, alpha):
        self.agent_id = agent_id
        self.link = link
        self.payoff = 0
        self.ave_payoff = 0
        self.actions = actions
        self.contribution = contribution
        self.h_value = np.zeros(len(actions))
        self.time = 1
        self.alpha = alpha
        self.strategy = np.zeros(len(actions))

    def initialize_strategy(self):
        sum_value = 0
        for i in range(len(self.actions)):
            sum_value += math.e ** self.h_value[i]
        for i in range(len(self.actions)):
            self.strategy[i] = math.e ** self.h_value[i] / sum_value

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

    def get_payoff(self):
        return self.payoff

    def get_ave_payoff(self):
        return self.ave_payoff

    def get_h_value(self):
        return self.h_value

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

    def update_time(self):
        self.time += 1

    def update_ave_payoff(self):
        self.ave_payoff = self.ave_payoff + 1 / self.time * self.payoff

    def update_h_value(self, a_i):
        for i in range(len(self.actions)):
            if a_i == self.actions[i]:
                self.h_value[i] = self.h_value[i] + self.alpha * (self.payoff - self.ave_payoff) * (1 - self.strategy[i])
            else:
                self.h_value[i] = self.h_value[i] - self.alpha * (self.payoff - self.ave_payoff) * self.strategy[i]

    def update_strategy(self):
        sum_value = 0
        for i in range(len(self.actions)):
            sum_value += math.e ** self.h_value[i]
        for i in range(len(self.actions)):
            self.strategy[i] = math.e ** self.h_value[i] / sum_value


def initialize_population(popu_size, adj_link, edge, actions, contribution, alpha):
    popu = []
    for i in range(popu_size):
        popu.append(Agent(i, adj_link[i], actions, contribution, alpha))
    for i in range(popu_size):
        popu[i].initialize_strategy()
        popu[i].update_time()
    return popu


def evolution_one_step(popu, edge, r, s, t, p):
    total_num = len(popu)
    for i in range(total_num):
        popu[i].set_payoff(0)
    a_l = [0 for _ in range(total_num)]
    c_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        print(popu[i].get_strategy())
        a_l[i] = np.random.choice(actions, p = popu[i].get_strategy())
    for i in range(total_num):
        if a_l[i] == 1:
            c_l[i] = popu[i].get_contribution()
    for pair in edge:
        ind_x = pair[0]
        ind_y = pair[1]
        p_x, p_y = pdd_game(a_l[ind_x], a_l[ind_y], r, s, t, p)
        popu[ind_x].add_payoff(p_x)
        popu[ind_y].add_payoff(p_y)
    for i in range(total_num):
        popu[i].update_ave_payoff()
        popu[i].update_h_value(a_l[i])
        popu[i].update_strategy()
        popu[i].update_time()
    return popu


def run(popu_size, contribution, adj_link, edge, actions, run_time, sample_time, alpha, r, s, t, p):
    popu = initialize_population(popu_size, adj_link, edge, actions, contribution, alpha)
    total_num = len(popu)
    for _ in range(run_time):
        popu = evolution_one_step(popu, edge, r, s, t, p)
    sample_strategy = []
    for _ in range(sample_time):
        popu = evolution_one_step(popu, edge, r, s, t, p)
        strategy_frac = np.zeros(len(actions))
        for i in range(total_num):
            strategy_frac += popu[i].get_strategy
        strategy_frac = strategy_frac / total_num
        sample_strategy.append(strategy_frac)
    return np.mean(sample_strategy, axis=0)


if __name__ == '__main__':
    popu_size = 100
    c = 1.0
    run_time = 1000
    sample_time = 20
    init_num = 5
    r = 3; s = 0; t = 5; p = 1
    actions = gen_actions()
    alpha = 0.1
    result_l = []
    for _ in range(init_num):
        adj, edge = generate_well_mixed_network(popu_size)
        sample_result = run(popu_size, c, adj, edge, actions, run_time, sample_time, alpha, r, s, t, p)
        print(sample_result)
        result_l.append(sample_result)
    print(np.mean(result_l, axis=0))
