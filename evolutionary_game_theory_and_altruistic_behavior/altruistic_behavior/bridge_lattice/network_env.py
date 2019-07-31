from bridge_lattice.social_distance_network import *

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


def generate_social_distance_network(structure, mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha, beta):
    if structure == "network_my_type":
        g_network = generate_network_my_type_connected(mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha)
        if g_network != None:
            adj_array = nx.to_numpy_array(g_network)
            adj_link = []
            for i in range(adj_array.shape[0]):
                adj_link.append(list(np.where(adj_array[i] == 1)[0]))
            population_num = group_size * (group_base ** (group_length - 1))
            g_edge = nx.Graph()
            for i in range(len(adj_link)):
                for j in range(len(adj_link[i])):
                    g_edge.add_edge(i, adj_link[i][j])
            return adj_link, population_num, g_edge.edges()
        else:
            return "Can not build a connected network"
    elif structure == "network_in_paper":
        g_network = generate_network_in_paper_connected(mul_dimen, degree, group_size, group_base, group_length, alpha, beta)
        if g_network != None:
            adj_array = nx.to_numpy_array(g_network)
            adj_link = []
            for i in range(adj_array.shape[0]):
                adj_link.append(list(np.where(adj_array[i] == 1)[0]))
            population_num = group_size * (group_base ** (group_length - 1))
            g_edge = nx.Graph()
            for i in range(len(adj_link)):
                for j in range(len(adj_link[i])):
                    g_edge.add_edge(i, adj_link[i][j])
            return adj_link, population_num, g_edge.edges()
        else:
            return "Can not build a connected network"
    elif structure == "network_hete":
        g_network = generate_hete_network_connected(mul_dimen, degree, group_size, group_base, group_length, alpha)
        if g_network != None:
            adj_array = nx.to_numpy_array(g_network)
            adj_link = []
            for i in range(adj_array.shape[0]):
                adj_link.append(list(np.where(adj_array[i] == 1)[0]))
            population_num = group_size * (group_base ** (group_length - 1))
            g_edge = nx.Graph()
            for i in range(len(adj_link)):
                for j in range(len(adj_link[i])):
                    g_edge.add_edge(i, adj_link[i][j])
            return adj_link, population_num, g_edge.edges()
        else:
            return "Can not build a connected network"
    else:
        return "No this type of structure"


if __name__ == '__main__':
    # adj_link_r, population_num_r, g_edge_r = generate_network(structure='2d_grid')
    # print(adj_link_r)
    group_size_r = 10
    group_base_r = 2
    group_length_r = 10
    mul_dimen_r = 5
    dimen_l_r = 1
    degree_r = 8
    alpha_r = -1
    beta_r = -1
    total_num_r = group_size_r * (group_base_r ** (group_length_r - 1))
    adj_link_r, population_num_r, g_edge_r = generate_social_distance_network("network_my_type", mul_dimen_r, degree_r, group_size_r,
                                                              group_base_r, group_length_r, dimen_l_r, alpha_r, beta_r)