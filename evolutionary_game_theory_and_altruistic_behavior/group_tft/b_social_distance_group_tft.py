import numpy as np
import pandas as pd
import random
import math
import argparse
import os
import datetime

from game_env import *
from network_env import *


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


def generate_opponent(total_num):
    opponent = [0 for _ in range(total_num)]
    for i in range(total_num):
        alter_ind = list(range(0, total_num))
        alter_ind.remove(i)
        opponent[i] = random.choice(alter_ind)
    return opponent


def initialize_strategy(total_num):
    ind_strategy = np.random.choice([0, 1, 2], total_num)
    return ind_strategy


def find_defectors(ind_action, pos_ind, group_base, group_length):
    position_num = group_base ** (group_length - 1)
    position_defectors = [[] for x in range(position_num)]
    for i in range(position_num):
        for j in pos_ind[i]:
            if ind_action[j] == 0:
                position_defectors.append(j)
    return position_defectors


def build_rep(ind_action, pos_ind, group_base, group_length):
    position_num = group_base ** (group_length - 1)
    position_rep = [0 for x in range(position_num)]
    for i in range(position_num):
        co_num = 0
        for j in pos_ind[i]:
            if ind_action[j] == 1:
                co_num += 1
        position_rep[i] = co_num / len(pos_ind[i])
    return position_rep


def run_game(ind_strategy, ind_rep, defect_param, group_size, group_base, group_length,
             total_num, ind_pos, pos_ind):
    if total_num != len(ind_pos):
        print('Error, the sum of individuals does not correspond to total number of individuals')
    old_ind_strategy = np.zeros(total_num)
    for i in range(total_num):
        old_ind_strategy[i] = ind_strategy[i]

    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        if ind_strategy[i] == 0:
            a_l[i] = 0
        elif ind_strategy[i] == 1:
            a_l[i] = 1

    opponent_play = generate_opponent(total_num)
    payoffs = np.zeros(total_num)
    for i in range(total_num):
        player_index = i
        opponent_index = opponent_play[i]
        if ind_strategy[player_index] == 2:
            if random.random() < ind_rep[opponent_index]:
                a_l[player_index] = 1
        if ind_strategy[opponent_index] == 2:
            if random.random() < ind_rep[player_index]:
                a_l[opponent_index] = 1
        payoffs_player, payoffs_opponent = play_b_game(a_l[player_index], a_l[opponent_index], defect_param)
        payoffs[player_index] += payoffs_player
        payoffs[opponent_index] += payoffs_opponent

    opponent_learn = generate_opponent(total_num)
    for i in range(total_num):
        player_index = i
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:
            potential_strategy = [0, 1, 2]
            potential_strategy.remove(old_ind_strategy[player_index])
            ind_strategy[player_index] = np.random.choice(potential_strategy)
        else:
            opponent_index = opponent_learn[player_index]
            t1 = 1 / (1 + math.e ** (10 * (payoffs[player_index] - payoffs[opponent_index])))
            t2 = random.random()
            if t2 < t1:
                ind_strategy[player_index] = old_ind_strategy[opponent_index]

    ind_rep_new = np.zeros(total_num)
    group_rep = build_rep(a_l, pos_ind, group_base, group_length)
