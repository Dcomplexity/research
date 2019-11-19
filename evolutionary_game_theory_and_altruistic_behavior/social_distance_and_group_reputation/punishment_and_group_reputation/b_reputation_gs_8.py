import numpy as np
import pandas as pd
import random
import math
import argparse
import os
import datetime


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


def donation_game(strategy_x, strategy_y, b):
    if strategy_x == 1 and strategy_y == 1:
        return b-1, b-1
    elif strategy_x == 1 and strategy_y == 0:
        return -1, b
    elif strategy_x == 0 and strategy_y == 1:
        return b, -1
    elif strategy_x == 0 and strategy_y == 0:
        return 0, 0
    else:
        return "Error: The strategy do not fit the conditions."


def pd_game_b(strategy_x, strategy_y, b):
    if strategy_x == 1 and strategy_y == 1:
        return 1, 1
    elif strategy_x == 1 and strategy_y == 0:
        return 0, b
    elif strategy_x == 0 and strategy_y == 1:
        return b, 0
    elif strategy_x == 0 and strategy_y == 0:
        return 0, 0
    else:
        return "Error: The strategy do not fit the conditions."


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


def build_rep(ind_strategy, pos_ind, group_base, group_length):
    position_num = group_base ** (group_length - 1)
    position_rep = [0 for x in range(position_num)]
    for i in range(position_num):
        co_num = 0
        for j in pos_ind[i]:
            if ind_strategy[j] == 1:
                co_num += 1
        position_rep[i] = co_num / len(pos_ind[i])
    return position_rep


def initialize_strategy(total_num):
    ind_strategy = np.random.choice([0, 1], total_num, p=[0.5, 0.5])
    return ind_strategy


def run_game(ind_strategy, ind_rep, defect_param, group_base, group_length,
             total_num, ind_pos, pos_ind, rt, rq):
    if total_num != len(ind_pos):
        print('Error, the sum of individuals does not correspond to total number of individuals')
    old_ind_strategy = np.zeros(total_num)
    for i in range(total_num):
        old_ind_strategy[i] = ind_strategy[i]

    opponent_play = generate_opponent(total_num)
    payoffs = np.zeros(total_num)
    for i in range(total_num):
        player_index = i
        player_strategy = ind_strategy[player_index]
        opponent_index = opponent_play[i]
        opponent_strategy = ind_strategy[opponent_index]
        rep_fre = 1 / (1 + rt * math.e ** -(ind_rep[player_index] * ind_rep[opponent_index] * rq))
        if np.random.random() < rep_fre:
            payoffs_i, payoffs_j = pd_game_b(player_strategy, opponent_strategy, defect_param)
            payoffs[player_index] += payoffs_i
            payoffs[opponent_index] += payoffs_j

    opponent_learn = generate_opponent(total_num)
    for i in range(total_num):
        player_index = i
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:
            potential_strategy = [0, 1]
            potential_strategy.remove(old_ind_strategy[i])
            ind_strategy[player_index] = np.random.choice(potential_strategy)
        else:
            opponent_index = opponent_learn[player_index]
            t1 = 1 / (1 + math.e ** (10 * (payoffs[player_index] - payoffs[opponent_index])))
            t2 = random.random()
            if t2 < t1:
                ind_strategy[player_index] = old_ind_strategy[opponent_index]

    ind_rep_new = np.zeros(total_num)
    group_rep = build_rep(old_ind_strategy, pos_ind, group_base, group_length)
    for i in range(total_num):
        ind_rep_new[i] = group_rep[ind_pos[i]]

    return ind_strategy, ind_rep_new


if __name__ == "__main__":
    group_size_r = 8
    group_base_r = 2
    group_length_r = 8
    rt_r = 100
    rq_r = 10
    total_num_r = group_size_r * (group_base_r ** (group_length_r - 1))
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results_old/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = dir_name + 'frac_co_b_reputation_gs_%s.csv' % group_size_r
    f = open(file_name, 'w')
    start_time = datetime.datetime.now()
    run_time = 80
    sample_time = 20
    rounds = 5
    results_r = []
    defect_param_r_list = []
    defect_param_r_generator = np.arange(1.0, 3.01, 0.1)
    for i in defect_param_r_generator:
        defect_param_r_list.append(round(i, 2))
    alpha_r = 1.0
    for defect_param_r in defect_param_r_list:
        print(defect_param_r)
        round_results_r = []
        for round_index in range(rounds):
            ind_strategy_r = initialize_strategy(total_num_r)
            ind_rep_r = [1.0 for _ in range(total_num_r)]
            for step_i in range(run_time):
                ind_strategy_r, ind_rep_new_r = run_game(ind_strategy_r, ind_rep_r, defect_param_r, group_base_r,
                                                         group_length_r, total_num_r, ind_pos_r, pos_ind_r, rt_r, rq_r)
                for i in range(total_num_r):
                    ind_rep_r[i] = ind_rep_r[i] + alpha_r * (ind_rep_new_r[i] - ind_rep_r[i])
            sample_strategy = []
            for step_i in range(sample_time):
                ind_strategy_r, ind_rep_new_r = run_game(ind_strategy_r, ind_rep_r, defect_param_r, group_base_r,
                                                         group_length_r, total_num_r, ind_pos_r, pos_ind_r, rt_r, rq_r)
                for i in range(total_num_r):
                    ind_rep_r[i] = ind_rep_r[i] + alpha_r * (ind_rep_new_r[i] - ind_rep_r[i])
                cal_strategy = np.zeros(2)
                for str_i in ind_strategy_r:
                    cal_strategy[str_i] += 1
                cal_strategy = cal_strategy / total_num_r
                sample_strategy.append(cal_strategy)
            round_results_r.append(np.mean(sample_strategy, axis=0))
        results_r.append(np.mean(round_results_r, axis=0))
    # m_index = pd.MultiIndex.from_product([alpha_r_list, defect_param_r_list], names=["alpha", "defect_param"])
    results_r_pd = pd.DataFrame(results_r, index=defect_param_r_list)
    results_r_pd.to_csv(f)
    f.close()
    end_time = datetime.datetime.now()
    print(results_r)
    print(end_time - start_time)