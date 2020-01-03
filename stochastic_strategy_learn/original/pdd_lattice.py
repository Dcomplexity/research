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


# Create the prisoners' dilemma game
def pd_game_b(a_x, a_y, b):
    if a_x == 1 and a_y == 1:
        p_x = 1; p_y = 1
    elif a_x == 1 and a_y == 0:
        p_x = 0; p_y = b
    elif a_x == 0 and a_y == 1:
        p_x = b; p_y = 0
    elif a_x == 0 and a_y == 0:
        p_x = 0; p_y = 0
    else:
        p_x = None; p_y = None
    return p_x, p_y


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
        popu.append(Agent(i, adj_link[i], np.random.choice([0, 1]), contribution))
    return popu


def evolution_one_step(popu, b):
    total_num = len(popu)
    for i in range(total_num):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(total_num)]  # action list
    c_l = [0 for _ in range(total_num)]  # contribution_list
    for i in range(total_num):
        if popu[i].get_strategy() == 0:
            a_l[i] = 0
            c_l[i] = 0
        else:
            a_l[i] = 1
            c_l[i] = popu[i].get_contribution()
    for pair in edge:
        ind_x = pair[0]
        ind_y = pair[1]
        p_x, p_y = pd_game_b(a_l[ind_x], a_l[ind_y], b)
        popu[ind_x].add_payoffs(p_x)
        popu[ind_y].add_payoffs(p_y)
    # Backup the strategy in this round
    for i in range(total_num):
        popu[i].set_ostrategy()
    # Update the strategy by imitating other's strategy
    for i in range(total_num):
        ind = popu[i]
        w1 = 0.01
        w2 = random.random()
        if w2 < w1:
            potential_strategy = [0, 1]
            potential_strategy.remove(ind.get_ostrategy())
            ind.set_strategy(np.random.choice(potential_strategy))
        else:
            ind_payoffs = ind.get_payoffs()
            j = np.random.choice(popu[i].get_link())
            opponent = popu[j]
            opponent_payoffs = opponent.get_payoffs()
            opponent_ostrategy = opponent.get_ostrategy()
            t1 = 1 / (1 + math.e ** (10.0 * (ind_payoffs - opponent_payoffs)))
            t2 = random.random()
            if t2 < t1:
                ind.set_strategy(opponent_ostrategy)
    return popu


def run(popu_size, contribution, adj_link, edge, run_time, sample_time, b):
    popu = initialize_population(popu_size, adj_link, edge, contribution)
    total_num = len(popu)
    for _ in range(run_time):
        popu = evolution_one_step(popu, b)
    sample_strategy = []
    for _ in range(sample_time):
        popu = evolution_one_step(popu, b)
        strategy_dist = [0 for x in range(2)]
        for i in range(total_num):
            strategy_dist[popu[i].get_strategy()] += 1
        strategy_dist = np.array(strategy_dist) / total_num
        sample_strategy.append(strategy_dist)
    return np.mean(sample_strategy, axis=0)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    popu_size = 1600
    c = 1.0
    run_time = 200
    sample_time = 20
    init_num = 5
    b_l = np.round(np.arange(0.0, 1.51, 0.1), 2)
    result_gamma = []
    for b_r in b_l:
        print(b_r)
        result = []
        for _ in range(init_num):
            adj, edge = generate_lattice(40, 40)
            sample_result = run(popu_size, c, adj, edge, run_time, sample_time, b_r)
            result.append(sample_result)
        result_gamma.append(np.mean(result, axis=0))
    result_gamma_pd = pd.DataFrame(result_gamma, index=b_l, columns=['d_frac', 'c_frac'])
    print(result_gamma_pd)
    file_name = './results/pgg_lattice.csv'
    result_gamma_pd.to_csv(file_name)
    end_time = datetime.datetime.now()
    print(end_time - start_time)



