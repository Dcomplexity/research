import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
import random

from network_build import *

def get_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta):
    G = generate_hete_network_connected(mul_dimen, degree, group_size, group_base, group_length, alpha)
    adj_array = nx.to_numpy_array(G)
    adj_link = []
    for i in range(adj_array.shape[0]):
        adj_link.append(list(np.where(adj_array[i] == 1)[0]))
    nodes = G.nodes
    edges = G.edges
    return adj_array, adj_link, nodes, edges


class Agent:
    def __init__(self, id, init_op, neighbor):
        self.id = id
        self.op = init_op
        self.old_op = init_op
        self.neighbor = neighbor

    def set_op(self, new_op):
        self.op = new_op

    def get_op(self):
        return self.op

    def get_old_op(self):
        return self.old_op

    def get_id(self):
        return self.id

    def backup(self):
        self.old_op = self.op

    def get_neighbor(self):
        return self.neighbor[:]




def initialize_population(group_size, group_base, group_length, mul_dimen, degree, alpha, beta):
    total_num = group_size * (group_base ** (group_length - 1))
    adj_array, adj_link, nodes, edges = get_network(mul_dimen, degree, group_size, group_base,
                                                    group_length, alpha, beta)
    population = []
    popu_num = len(nodes)
    for i in nodes:
        # if i / popu_num <= 0.5:
        #     population.append(Agent(i, i/popu_num + 0.5, adj_link[i]))
        # else:
        #     population.append(Agent(i, i/popu_num - 0.5, adj_link[i]))
        population.append(Agent(i, (i+popu_num/2)%popu_num/popu_num, adj_link[i]))
    return population


def run(popu, bound, iter_num):
    popu_num = len(popu)
    op_history = [[] for _ in range(popu_num)]
    for _ in range(iter_num):
        for i in range(popu_num):
            i_op = popu[i].get_old_op()
            op_history[i].append(i_op)
            neighbors = popu[i].get_neighbor()
            neighbors.append(i)
            op_in_bound = []
            for j in neighbors:
                j_op = popu[j].get_old_op()
                if abs(i_op - j_op) < bound or (1.0 - abs(i_op - j_op)) < bound:
                # if abs(i_op - j_op) < bound:
                    op_in_bound.append(j_op)
            new_op = np.mean(op_in_bound)
            popu[i].set_op(new_op)
        for i in range(popu_num):
            popu[i].backup()
    return op_history


if __name__ == '__main__':
    group_size_r = 50
    group_base_r = 2
    group_length_r = 6
    mul_dimen_r = 10
    degree_r = 20
    alpha_r = 2
    beta_r = 2
    total_num_r = group_size_r * (group_base_r ** (group_length_r - 1))
    popu_r = initialize_population(group_size_r, group_base_r, group_length_r, mul_dimen_r, degree_r, alpha_r, beta_r)
    op_history_r = run(popu_r, 0.3, 50)
    op_history_pd = pd.DataFrame(op_history_r)
    plt.figure()
    op_history_pd.T.plot(legend=False)
    plt.show()
    print(op_history_pd)