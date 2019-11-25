import numpy as np
import pandas as pd
import random
import math
import argparse
import os
import datetime
import networkx as nx


class SocialStructure():
    def __init__(self, group_size, group_base, group_length, total_num):
        self.group_size = group_size
        self.group_base = group_base
        self.group_length = group_length
        self.total_num = total_num
        self.group_num = self.group_base ** (self.group_length - 1)
        if self.total_num != self.group_size * (self.group_base ** (self.group_length - 1)):
            print("Error: The total num of individuals does not correspond to the social structure")


    def build_social_structure(self):
        ind_pos = [0 for x in range(self.total_num)]
        pos_ind = [[] for x in range(self.group_num)]
        for i in range(self.group_num):
            for j in range(i*self.group_size, (i+1)*self.group_size):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(struc_group_size, struc_group_base, struc_group_length):
    struc_total_num = struc_group_size * (struc_group_base ** (struc_group_length - 1))
    struc_social_structure = SocialStructure(struc_group_size, struc_group_base, struc_group_length, struc_total_num)
    struc_ind_pos, struc_pos_ind = struc_social_structure.build_social_structure()
    return struc_ind_pos, struc_pos_ind


def generate_distance_prob(group_length, group_size, reg_param):
    distance_dist = np.zeros(group_length)
    # If there is only one individual in a group, individual should not interact with himself.
    if group_size == 1:
        for k in range(1, group_length):
            distance_dist[k] = math.e ** (reg_param * (k + 1))
    else:
        for k in range(group_length):
            distance_dist[k] = math.e ** (reg_param * (k + 1))
    distance_dist = distance_dist / np.sum(distance_dist)
    return distance_dist


def get_cover_position(group_length, now_position, distance_prob):
    potential_pos = []
    distance = np.random.choice(group_length, 1, p=distance_prob)[0] + 1
    # print(distance)
    if distance == 1:
        potential_pos.append(now_position)
    else:
        pos_temp = 2 ** (distance - 1)
        for k in range(0, pos_temp):
            potential_pos.append((now_position // pos_temp) * pos_temp + k)
    return potential_pos


def get_neigh_position(group_length, now_position, distance_prob):
    potential_pos =[]
    distance = np.random.choice(group_length, 1, p=distance_prob)[0] + 1
    # print(distance)
    if distance == 1:
        potential_pos.append(now_position)
    else:
        pos_temp = 2 ** (distance - 1)
        if now_position % pos_temp < pos_temp // 2:
            for k in range(pos_temp // 2, pos_temp):
                potential_pos.append((now_position // pos_temp) * pos_temp + k)
        else:
            for k in range(0, pos_temp // 2):
                potential_pos.append((now_position // pos_temp) * pos_temp + k)
    return potential_pos


# def new_cover_structure(pos_ind, group_size, dimension_distance):
#     # dimension_distance means the difference between dimensions
#     new_space = 2 ** (dimension_distance - 1)
#     len_pos = len(pos_ind)
#     new_pos_ind = []
#     for i in range(0, len_pos, new_space):
#         new_pos_pool = []
#         for j in range(i, i+new_space):
#             new_pos_pool.extend(pos_ind[j])
#         random.shuffle(new_pos_pool)
#         for k in range(len(new_pos_pool)//group_size):
#             new_pos_ind.append(new_pos_pool[k * group_size : (k + 1) * group_size])
#     return new_pos_ind

def new_cover_structure(pos_ind, group_length, group_size, reg_param):
    distance_prob = generate_distance_prob(group_length, group_size, reg_param)
    dimension_distance = np.random.choice(group_length, 1, p=distance_prob)[0] + 1
    new_space = 2 ** (dimension_distance - 1)
    len_pos = len(pos_ind)
    new_pos_ind = []
    for i in range(0, len_pos, new_space):
        new_pos_pool = []
        for j in range(i, i+new_space):
            new_pos_pool.extend(pos_ind[j])
        random.shuffle(new_pos_pool)
        for k in range(len(new_pos_pool)//group_size):
            new_pos_ind.append(new_pos_pool[k * group_size: (k + 1) * group_size])
    return new_pos_ind


def new_neigh_structure(pos_ind, ind_pos, group_length, group_size, reg_param):
    len_pos = len(pos_ind)
    len_ind = len(ind_pos)
    new_pos_ind = [[] for _ in range(len_pos)]
    distance_prob = generate_distance_prob(group_length, group_size, reg_param)
    for i in range(len_ind):
        now_position = ind_pos[i]
        potential_pos = get_neigh_position(group_length, now_position, distance_prob)
        new_pos = random.choice(potential_pos)
        new_pos_ind[new_pos].append(i)
    return new_pos_ind


def new_hete_structure(pos_ind, group_size, group_length):
    new_pos_ind = []
    start_pos = 0
    for i in range(0, group_length-1):
        if i == 0:
            new_pos_ind.append(pos_ind[0])
            new_pos_ind.append(pos_ind[1])
            start_pos = 2
        else:
            interval_space = 2 ** i
            end_pos = start_pos + interval_space
            new_pos_pool = []
            for j in range(start_pos, end_pos):
                new_pos_pool.extend(pos_ind[j])
            random.shuffle(new_pos_pool)
            for k in range(len(new_pos_pool) // group_size):
                new_pos_ind.append(new_pos_pool[k * group_size : (k + 1) * group_size])
            start_pos = end_pos
    return new_pos_ind


# def build_multi_pos_ind(mul_dimen, pos_ind, group_size, dimension_distance):
#     pos_ind_mul = []
#     for _ in range(mul_dimen):
#         pos_ind_mul.append(new_cover_structure(pos_ind, group_size, dimension_distance))
#     return pos_ind_mul


def build_multi_pos_ind(mul_dimen, pos_ind, group_length, group_size, beta):
    pos_ind_mul = []
    for _ in range(mul_dimen):
        pos_ind_mul.append(new_cover_structure(pos_ind, group_length, group_size, beta))
    return pos_ind_mul


def build_multi_ind_pos(pos_ind_mul, total_num):
    ind_pos_mul = [[0 for i in range(total_num)] for j in range(len(pos_ind_mul))]
    for i in range(len(pos_ind_mul)):
        for j in range(len(pos_ind_mul[i])):
            for k in pos_ind_mul[i][j]:
                ind_pos_mul[i][k] = j
    return ind_pos_mul


def pick_individual(ind_self, positions, pos_ind):
    potential_ind = []
    for i in positions:
        for j in pos_ind[i]:
            potential_ind.append(j)
    # remove the individual itself from the potential opponents list
    if ind_self in potential_ind:
        potential_ind.remove(ind_self)
    ind_index = np.random.choice(potential_ind, 1)[0]
    return int(ind_index)


def average_degree(graph, total_num):
    degree_hist = nx.degree_histogram(graph)
    degree_sum = 0
    for i in range(len(degree_hist)):
        degree_sum += degree_hist[i] * i
    return degree_sum / total_num


# def generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha):
#     total_num = group_size * (group_base ** (group_length - 1))
#     edge_num = degree * total_num / 2
#     ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
#     pos_ind_mul = []
#     for i in range(mul_dimen):
#         pos_ind_mul.append(new_cover_structure(pos_ind, group_size, dimen_l))
#     ind_pos_mul = build_multi_ind_pos(pos_ind_mul, total_num)
#     distance_prob = generate_distance_prob(group_length, group_size, alpha)
#     adj_link = {}
#     edge_count = 0
#     for i in range(total_num):
#         adj_link[i] = []
#     while edge_count < edge_num:
#         i_chosen = np.random.choice(total_num)
#         mul_chosen = np.random.choice(mul_dimen)
#         ind_pos_chosen = ind_pos_mul[mul_chosen]
#         pos_ind_chosen = pos_ind_mul[mul_chosen]
#         i_pos = ind_pos_chosen[i_chosen]
#         potential_pos = get_cover_position(group_length, i_pos, distance_prob)
#         potential_ind = pick_individual(i_chosen, potential_pos, pos_ind_chosen)
#         if potential_ind not in adj_link[i_chosen]:
#             adj_link[i_chosen].append(potential_ind)
#             adj_link[potential_ind].append(i_chosen)
#             edge_count += 1
#     return adj_link


def generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta):
    total_num = group_size * (group_base ** (group_length - 1))
    edge_num = degree * total_num / 2
    ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
    pos_ind_mul = []
    for i in range(mul_dimen):
        pos_ind_mul.append(new_cover_structure(pos_ind, group_length, group_size, beta))
    ind_pos_mul = build_multi_ind_pos(pos_ind_mul, total_num)
    distance_prob = generate_distance_prob(group_length, group_size, alpha)
    adj_link = {}
    edge_count = 0
    for i in range(total_num):
        adj_link[i] = []
    while edge_count < edge_num:
        i_chosen = np.random.choice(total_num)
        mul_chosen = np.random.choice(mul_dimen)
        ind_pos_chosen = ind_pos_mul[mul_chosen]
        pos_ind_chosen = pos_ind_mul[mul_chosen]
        i_pos = ind_pos_chosen[i_chosen]
        potential_pos = get_cover_position(group_length, i_pos, distance_prob)
        potential_ind = pick_individual(i_chosen, potential_pos, pos_ind_chosen)
        if potential_ind not in adj_link[i_chosen]:
            adj_link[i_chosen].append(potential_ind)
            adj_link[potential_ind].append(i_chosen)
            edge_count += 1
    return adj_link


def generate_neigh_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta):
    total_num = group_size * (group_base ** (group_length - 1))
    edge_num = degree * total_num / 2
    ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
    pos_ind_mul = []
    for i in range(mul_dimen):
        pos_ind_mul.append(new_neigh_structure(pos_ind, ind_pos, group_length, group_size, beta))
    ind_pos_mul = build_multi_ind_pos(pos_ind_mul, total_num)
    distance_prob = generate_distance_prob(group_length, group_size, alpha)
    adj_link = {}
    edge_count = 0
    for i in range(total_num):
        adj_link[i] = []
    while edge_count < edge_num:
        i_chosen = np.random.choice(total_num)
        mul_chosen = np.random.choice(mul_dimen)
        ind_pos_chosen = ind_pos_mul[mul_chosen]
        pos_ind_chosen = pos_ind_mul[mul_chosen]
        i_pos = ind_pos_chosen[i_chosen]
        potential_pos = get_neigh_position(group_length, i_pos, distance_prob)
        potential_ind = pick_individual(i_chosen, potential_pos, pos_ind_chosen)
        if potential_ind not in adj_link[i_chosen]:
            adj_link[i_chosen].append(potential_ind)
            adj_link[potential_ind].append(i_chosen)
            edge_count += 1
    return adj_link


def generate_hete_network(mul_dimen, degree, group_size, group_base, group_length, alpha):
    total_num = group_size * (group_base ** (group_length - 1))
    edge_num = degree * total_num / 2
    ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
    pos_ind_mul = []
    for i in range(mul_dimen):
        pos_ind_mul.append(new_hete_structure(pos_ind, group_size, group_length))
    ind_pos_mul = build_multi_ind_pos(pos_ind_mul, total_num)
    distance_prob = generate_distance_prob(group_length, group_size, alpha)
    adj_link = {}
    edge_count = 0
    for i in range(total_num):
        adj_link[i] = []
    while edge_count < edge_num:
        i_chosen = np.random.choice(total_num)
        mul_chosen = np.random.choice(mul_dimen)
        ind_pos_chosen = ind_pos_mul[mul_chosen]
        pos_ind_chosen = pos_ind_mul[mul_chosen]
        i_pos = ind_pos_chosen[i_chosen]
        potential_pos = get_cover_position(group_length, i_pos, distance_prob)
        potential_ind = pick_individual(i_chosen, potential_pos, pos_ind_chosen)
        if potential_ind not in adj_link[i_chosen]:
            adj_link[i_chosen].append(potential_ind)
            adj_link[potential_ind].append(i_chosen)
            edge_count += 1
    return adj_link


# def generate_cover_network_connected(mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha):
#     adj_link = generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha)
#     G = nx.Graph(adj_link)
#     gen_num = 0
#     while not nx.is_connected(G):
#         print(gen_num)
#         adj_link = generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, dimen_l, alpha)
#         G = nx.Graph(adj_link)
#         gen_num += 1
#         if gen_num > 50:
#             print("Not found connected network")
#             return None
#     return G


def generate_cover_network_connected(mul_dimen, degree, group_size, group_base, group_length, alpha, beta):
    adj_link = generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta)
    G = nx.Graph(adj_link)
    gen_num = 0
    while not nx.is_connected(G):
        print(gen_num)
        adj_link = generate_cover_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta)
        G = nx.Graph(adj_link)
        gen_num += 1
        if gen_num > 50:
            print("Not found connected network")
            return None
    return G


def generate_neigh_network_connected(mul_dimen, degree, group_size, group_base, group_length, alpha, beta):
    adj_link = generate_neigh_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta)
    G = nx.Graph(adj_link)
    gen_num = 0
    while not nx.is_connected(G):
        print(gen_num)
        adj_link = generate_neigh_network(mul_dimen, degree, group_size, group_base, group_length, alpha, beta)
        G = nx.Graph(adj_link)
        gen_num += 1
        if gen_num > 50:
            print("Not found connected network")
            return None
    return G


def generate_hete_network_connected(mul_dimen, degree, group_size, group_base, group_length, alpha):
    adj_link = generate_hete_network(mul_dimen, degree, group_size, group_base, group_length, alpha)
    G = nx.Graph(adj_link)
    gen_num = 0
    while not nx.is_connected(G):
        print(gen_num)
        adj_link = generate_hete_network(mul_dimen, degree, group_size, group_base, group_length, alpha)
        G = nx.Graph(adj_link)
        gen_num += 1
        if gen_num > 50:
            print("Not found connected network")
            return None
    return G


def generate_er_random_connected(degree, group_size, group_base, group_length):
    population_num = group_size * (group_base ** (group_length - 1))
    G = nx.erdos_renyi_graph(population_num, p=degree/(population_num-1))
    gen_num = 0
    while not nx.is_connected(G):
        print(gen_num)
        G = nx.erdos_renyi_graph(population_num, p=degree/(population_num-1))
        gen_num += 1
        if gen_num > 50:
            print("Not found connected network")
            return None
    return G


if __name__ == '__main__':
    group_size_r = 100
    group_base_r = 2
    group_length_r = 6
    mul_dimen_r = 10
    degree_r = 10
    alpha_r = -2
    beta_r = -2
    total_num_r = group_size_r * (group_base_r ** (group_length_r - 1))
    # G_r = generate_cover_network_connected(mul_dimen_r, degree_r, group_size_r,
    #                                        group_base_r, group_length_r, alpha_r, beta_r)
    G_r = generate_hete_network_connected(mul_dimen_r, degree_r, group_size_r, group_base_r, group_length_r, alpha_r)
    G = G_r
    print(G.edges())
    if G != None:
        cc = nx.average_clustering(G)
        ER_G = nx.erdos_renyi_graph(total_num_r, p=degree_r/(total_num_r-1))
        cc0 = nx.average_clustering(ER_G)
        print(cc/cc0)
        print(average_degree(G, total_num_r))
    else:
        print("Not connected")
    # distance_prob_r = generate_distance_prob(group_length_r, group_size_r, alpha_r)
    # print(distance_prob_r)
    # potential_pos_r = get_neigh_position(group_length_r, 0, distance_prob_r)
    # print(potential_pos_r)