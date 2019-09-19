import pandas as pd
import numpy as np
import random
import math

class SocialStructure():
    def __init__(self, g_s, g_b, g_l, t_n):
        self.g_s = g_s  # group_size
        self.g_b = g_b  # group_base
        self.g_l = g_l  # group_length
        self.t_n = t_n  # total_num
        self.g_n = self.g_b ** (self.g_l - 1)
        if self.t_n != self.g_s * self.g_n:
            print("Error: The total num of individuals does not correspond to the social structure")

    def build_social_structure(self):
        ind_pos = [0 for x in range(self.t_n)]
        pos_ind = [[] for x in range(self.g_n)]
        for i in range(self.g_n):
            for j in range(i*self.g_s, (i+1)*self.g_s):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def build_structure(g_s, g_b, g_l):
    t_n = g_s * (g_b ** (g_l - 1))
    s_s = SocialStructure(g_s, g_b, g_l, t_n)
    ind_pos, pos_ind = s_s.build_social_structure()
    return ind_pos, pos_ind


def generate_distance_prob(g_l, g_s, reg_p):
    distance_dist = np.zeros(g_l)
    # If there is only one individual in a group, individual should not interact with himself.
    if g_s == 1:
        for k in range(1, g_l):
            distance_dist[k] = math.e ** (reg_p * (k + 1))
    else:
        for k in range(g_l):
            distance_dist[k] = math.e ** (reg_p * (k + 1))
    distance_dist = distance_dist / np.sum(distance_dist)
    return distance_dist


# get the potential positions including itself based on distance probability
def get_cover_position(g_l, now_pos, dis_prob):
    poten_pos = []
    distance = np.random.choice(g_l, 1, p=dis_prob)[0] + 1
    if distance == 1:
        poten_pos.append(now_pos)
    else:
        pos_num = 2 ** (distance - 1)
        for k in range(0, pos_num):
            poten_pos.append((now_pos // pos_num) * pos_num + k)
    return poten_pos


# get the potential positions not including itself based on distance probability
def get_neigh_position(g_l, now_pos, dis_prob):
    poten_pos = []
    distance = np.random.choice(g_l, 1, p=dis_prob)[0] + 1
    if distance == 1:
        poten_pos.append(now_pos)
    else:
        pos_num = 2 ** (distance - 1)
        if now_pos % pos_num < pos_num // 2:
            for k in range(pos_num // 2, pos_num):
                poten_pos.append((now_pos // pos_num) * pos_num + k)
        else:
            for k in range(0, pos_num // 2):
                poten_pos.append((now_pos // pos_num) * pos_num + k)
    return poten_pos


def new_cover_structure(pos_ind, g_l, g_s, reg_p, dimen_dis=None):
    if not dimen_dis:
        dis_prob = generate_distance_prob(g_l, g_s, reg_p)
        dimen_dis = np.random.choice(g_l, 1, p=dis_prob)[0] + 1
    new_space = 2 ** (dimen_dis - 1)
    len_pos = len(pos_ind)
    new_pos_ind = []
    for i in range(0, len_pos, new_space):
        new_pos_pool = []
        for j in range(i, i+new_space):
            new_pos_pool.extend(pos_ind[j])
        random.shuffle(new_pos_pool)
        for k in range(len(new_pos_pool)//g_s):
            new_pos_ind.append(new_pos_pool[k * g_s: (k + 1) * g_s])
    return new_pos_ind


def new_same_structure(pos_ind):
    return pos_ind


def new_neigh_structure(pos_ind, ind_pos, g_l, g_s, reg_p, dimen_dis=None):
    len_pos = len(pos_ind)
    len_ind = len(ind_pos)
    new_pos_ind = [[] for _ in range(len_pos)]
    dis_prob = generate_distance_prob(g_l, g_s, reg_p)
    for i in range(len_ind):
        now_pos = ind_pos[i]
        poten_pos = get_neigh_position(g_l, now_pos, dis_prob)
        new_pos = random.choice(poten_pos)
        new_pos_ind[new_pos].append(i)
    return new_pos_ind


def new_hete_structure(pos_ind, g_s, g_l):
    new_pos_ind = []
    start_pos = 0
    for i in range(0, g_l - 1):
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
            for k in range(len(new_pos_pool) // g_s):
                new_pos_ind.append(new_pos_pool[k * g_s : (k + 1) * g_s])
            start_pos = end_pos
    return new_pos_ind


def build_multi_pos_ind(mul_dimen, pos_ind, g_l, g_s, beta):
    pos_ind_mul = []
    for _ in range(mul_dimen):
        pos_ind_mul.append(new_cover_structure(pos_ind, g_l, g_s, beta))
    return pos_ind_mul


def build_multi_same_pos_ind(mul_dimen, pos_ind):
    pos_ind_mul = []
    for _ in range(mul_dimen):
        pos_ind_mul.append(pos_ind)
    return pos_ind_mul


def build_multi_ind_pos(pos_ind_mul, t_n):
    ind_pos_mul = [[0 for i in range(t_n)] for j in range(len(pos_ind_mul))]
    for i in range(len(pos_ind_mul)):
        for j in range(len(pos_ind_mul[i])):
            for k in pos_ind_mul[i][j]:
                ind_pos_mul[i][k] = j
    return ind_pos_mul


if __name__ == '__main__':
    group_size = 16; group_base = 2; group_length = 5; mul_dimen = 5; beta = -2
    total_number = group_size * (group_base ** (group_length - 1))
    ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
    pos_ind_multi = build_multi_pos_ind(mul_dimen, pos_ind, group_length, group_size, beta)
    ind_pos_multi = build_multi_ind_pos(pos_ind_multi, total_number)
    print(pos_ind_multi)



