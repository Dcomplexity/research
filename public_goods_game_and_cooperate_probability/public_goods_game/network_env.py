import networkx as nx
import numpy as np

def generate_network(structure, xdim=40, ydim=40, nodes_num=1600, edge_num=2):
    if structure == "2d_grid":
        g_network = nx.grid_2d_graph(xdim, ydim, periodic=True)
        adj_array = nx.to_numpy_array(g_network)
        adj_link = []
        population_num = 0
        for i in range(adj_array.shape[0]):
            adj_link.append(list(np.where(adj_array[i] == 1)[0]))
        population_num = xdim * ydim
        g_edge = nx.Graph()
        for i in range(len(adj_link)):
            for j in range(len(adj_link[i])):
                g_edge.add_edge(i, adj_link[i][j])
        return adj_link, population_num, g_edge.edges()
    elif structure == "triangular_lattice":
        g_network = nx.triangular_lattice_graph(xdim, ydim * 2, periodic=True)
        adj_array = nx.to_numpy_array(g_network)
        adj_link = []
        population_num = 0
        for i in range(adj_array.shape[0]):
            adj_link.append(list(np.where(adj_array[i] == 1)[0]))
        population_num = xdim * ydim
        g_edge = nx.Graph()
        for i in range(len(adj_link)):
            for j in range(len(adj_link[i])):
                g_edge.add_edge(i, adj_link[i][j])
        return adj_link, population_num, g_edge.edges()
    elif structure == "ba_graph":
        g_network = nx.barabasi_albert_graph(n=nodes_num, m=edge_num)
        adj_array = nx.to_numpy_array(g_network)
        adj_link = []
        population_num = 0
        for i in range(adj_array.shape[0]):
            adj_link.append(list(np.where(adj_array[i] == 1)[0]))
        population_num = nodes_num
        g_edge = nx.Graph()
        for i in range(len(adj_link)):
            for j in range(len(adj_link[i])):
                g_edge.add_edge(i, adj_link[i][j])
        return adj_link, population_num, g_edge.edges()
    else:
        return "No this type of structure"


def generate_er_random_connected(degree, nodes_num):
    if degree == 6:
        # seed=101 can ensure a connected network
        g_network = nx.fast_gnp_random_graph(nodes_num, p=degree/(nodes_num-1), seed=101)
    else:
        g_network = nx.fast_gnp_random_graph(nodes_num, p=degree/(nodes_num-1))
    gen_num = 0
    while not nx.is_connected(g_network):
        rnd_num = np.random
        g_network = nx.fast_gnp_random_graph(nodes_num, p=degree/(nodes_num-1), seed=rnd_num)
        gen_num += 1
        if gen_num > 100:
            print("Not found connected network")
            return None
    adj_array = nx.to_numpy_array(g_network)
    adj_link = []
    for i in range(adj_array.shape[0]):
        adj_link.append(list(np.where(adj_array[i] == 1)[0]))
    population_num = nodes_num
    g_edge = nx.Graph()
    for i in range(len(adj_link)):
        for j in range(len(adj_link[i])):
            g_edge.add_edge(i, adj_link[i][j])
    return adj_link, population_num, g_edge.edges()


if __name__ == '__main__':
    adj_link_r, population_num_r, g_edge_r = generate_network(structure='2d_grid')
    print(adj_link_r)