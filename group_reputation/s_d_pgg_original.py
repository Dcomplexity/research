import numpy as np
import pandas as pd
import random
import math
import argparse
import os
import datetime

from game import *

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


def initial_action(t_n):
    init_a = np.random.choice([0, 1], t_n, p = [0.5, 0.5])
    return init_a


def game_one_round(a_l, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    pos_n = len(pos_ind)
    a_l_old = a_l[:]
    p_l = [0 for _ in range(ind_n)]
    for pos in range(pos_n):
        g_inds = pos_ind[pos]
        g_inds_n = len(g_inds)
        g_a = []
        for i in range(g_inds_n):
            g_a.append(a_l[g_inds[i]])
        g_p = pgg_game(g_a, gamma)
        for i in range(g_inds_n):
            p_l[g_inds[i]] += g_p[i]
    for ind in range(ind_n):
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:  # mutation
            a_l[ind] = 1 - a_l[ind]
        else:
            while True:
                oppon = random.choice(range(ind_n))
                if oppon != ind:
                    break
            ind_p = p_l[ind]
            oppon_p = p_l[oppon]
            t1 = 1 / (1 + math.e ** (10 * (ind_p - oppon_p)))
            t2 = random.random()
            if t2 < t1:
                a_l[ind] = a_l_old[oppon]
    return a_l

def run_game(run_time, gamma, ind_pos, pos_ind):
    ind_n = len(ind_pos)
    a_l = initial_action(ind_n)
    for step in range(run_time):
        a_l = game_one_round(a_l, gamma, ind_pos, pos_ind)
    return a_l

def evaluation(eval_time, gamma, ind_pos, pos_ind, a_l):
    ind_n = len(ind_pos)
    a_frac = 0
    for step in range(eval_time):
        a_l = game_one_round(a_l, gamma, ind_pos, pos_ind)
        a_frac = step / (step + 1) * a_frac + 1 / (step + 1) * np.mean(a_l)
    return a_frac


if __name__ == '__main__':
    group_size = 4; group_base = 2; group_length = 4
    ind_pos, pos_ind = build_structure(group_size, group_base, group_length)
    run_time = 100; gamma = 0.3; eval_time = 10
    init_time = 10
    r_a_frac = 0
    for i in range(init_time):
        a_l = run_game(run_time, gamma, ind_pos, pos_ind)
        a_frac = evaluation(eval_time, gamma, ind_pos, pos_ind, a_l)
        r_a_frac = i / (i + 1) * r_a_frac + 1 / (i + 1) * a_frac
    print(r_a_frac)
