import numpy as np
from scipy.stats import binom
import random
import math
import pandas as pd
import networkx as nx

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
                to_link_node = np.random.choice(t)
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

if __name__ == '__main__':
    g_n = 30; init_time = 100
    for r_value in [2, 2.2, 2.5, 3, 5, 7, 10]:
        print(r_value)
        adj_matrix = np.zeros((g_n, g_n))
        for i in range(init_time):
            price_graph = price_model(g_n, 2, r_value)
            adj_matrix += nx.adjacency_matrix(price_graph)
        adj_matrix = adj_matrix / init_time
        adj_matrix = np.array(adj_matrix)
        adj_matrix_pd = pd.DataFrame(adj_matrix)
        csv_file_name = './results/stochastic_price_model_%.1f.csv' % r_value
        adj_matrix_pd.to_csv(csv_file_name, index=None, columns=None)
        print(adj_matrix_pd)

