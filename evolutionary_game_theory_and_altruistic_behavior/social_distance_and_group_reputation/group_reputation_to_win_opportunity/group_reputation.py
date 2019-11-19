import datetime
import math
import os
import random

import numpy as np
import pandas as pd


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
            for j in range(i * self.group_size, (i + 1) * self.group_size):
                ind_pos[j] = i
                pos_ind[i].append(j)
        return ind_pos, pos_ind


def donation_game(strategy_x, strategy_y, b):
    if strategy_x == 1 and strategy_y == 1:
        return b - 1, b - 1
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


def distance_prob(group_length, group_size, reg_param):
    distance_dist = np.zeros(group_length)
    if group_size == 1:
        for k in range(1, group_length):
            distance_dist[k] = math.e ** (reg_param * (k + 1))
    else:
        for k in range(group_length):
            distance_dist[k] = math.e ** (reg_param * (k + 1))
    distance_dist = distance_dist / np.sum(distance_dist)
    return distance_dist


def get_position(group_length, now_position, distance_prob,
                 position_rep):  # The set of position includes the group itself.
    potential_pos = []
    distance = np.random.choice(group_length, 1, p=distance_prob)[0] + 1
    if distance == 1:
        potential_pos.append(now_position)
    else:
        pos_temp = 2 ** (distance - 1)
        for k in range(0, pos_temp):
            potential_pos.append((now_position // pos_temp) * pos_temp + k)
    # pick one position from the potential positions based on group reputation
    potential_pos_rep = []
    for i in potential_pos:
        potential_pos_rep.append(position_rep[i])
    if np.sum(potential_pos_rep) == 0.0:
        potential_pos_rep = [1.0 / len(potential_pos_rep) for _ in
                             range(len(potential_pos_rep))]  # if the reputations of all group are 0, pick randomly
    else:
        potential_pos_rep = np.array(potential_pos_rep) / np.sum(potential_pos_rep)
    pos_chosen = np.random.choice(potential_pos, 1, p=potential_pos_rep)
    return potential_pos, pos_chosen


def pick_individual(ind_self, positions, pos_ind):
    potential_ind = []
    for i in positions:
        potential_ind.extend(pos_ind[i])
    while True:
        ind_index = int(random.choice(potential_ind))
        if ind_index != ind_self:
            break
    return ind_index


def pick_individual_randomly(ind_self, total_num):
    while True:
        ind_index = int(np.random.choice(total_num))
        if ind_index != ind_self:
            break
    return ind_index


# def pick_random_individual(total_num):
#     opponent = [0 for _ in range(total_num)]
#     for i in range(total_num):
#         alter_ind = list(range(0, total_num))
#         alter_ind.remove(i)
#         opponent[i] = random.choice(alter_ind)
#     return opponent


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


def run_game(ind_strategy, alpha, beta, group_rep, defect_param, group_size, group_base, group_length,
             total_num, ind_pos, pos_ind):
    if total_num != len(ind_pos):
        print("Error, the sum of individuals does not correspond to total number of individuals")
    old_ind_strategy = np.zeros(total_num)
    for i in range(total_num):
        old_ind_strategy[i] = ind_strategy[i]

    prob_play = distance_prob(group_length, group_size, alpha)
    prob_learn = distance_prob(group_length, group_size, beta)

    opponent_play = np.zeros(total_num, dtype=int)
    opponent_learn = np.zeros(total_num, dtype=int)
    payoffs = np.zeros(total_num)

    for i in range(total_num):
        now_position = ind_pos[i]
        potential_pos, pos_chosen = get_position(group_length, now_position, prob_play, group_rep)
        opponent_play[i] = pick_individual(i, pos_chosen, pos_ind)

    for i in range(total_num):
        now_position = ind_pos[i]
        potential_pos, pos_chosen = get_position(group_length, now_position, prob_learn, group_rep)
        # learn from an individuals who is in a group with high reputation
        # opponent_learn[i] = pick_individual(i, pos_chosen, pos_ind)
        # learn from an individuals within the distance
        # opponent_learn[i] = pick_individual(i, potential_pos, pos_ind)
        # learn from an individual picked randomly
        opponent_learn[i] = pick_individual_randomly(i, total_num)

    for i in range(total_num):
        player_index = i
        opponent_index = opponent_play[player_index]
        player_strategy = ind_strategy[player_index]
        opponent_strategy = ind_strategy[opponent_index]
        payoffs_i, payoffs_j = pd_game_b(player_strategy, opponent_strategy, defect_param)
        payoffs[player_index] += payoffs_i
        payoffs[opponent_index] += payoffs_j

    for i in range(total_num):
        player_index = i
        w1 = 0.01
        w2 = random.random()
        if w1 > w2:
            potential_strategy = [0, 1]
            potential_strategy.remove(old_ind_strategy[player_index])
            ind_strategy[player_index] = np.random.choice(potential_strategy)
        else:
            opponent_index = opponent_learn[player_index]
            t1 = 1 / (1 + math.e ** (10 * (payoffs[player_index] - payoffs[opponent_index])))
            t2 = random.random()
            if t2 < t1:
                ind_strategy[player_index] = old_ind_strategy[opponent_index]

    new_group_rep = build_rep(old_ind_strategy, pos_ind, group_base, group_length)

    return ind_strategy, new_group_rep


if __name__ == '__main__':
    group_size_r = 8
    group_base_r = 2
    group_length_r = 8
    group_num_r = group_base_r ** (group_length_r - 1)
    total_num_r = group_size_r * group_num_r
    ind_pos_r, pos_ind_r = build_structure(group_size_r, group_base_r, group_length_r)
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results_old/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = dir_name + 'frac_co_b_reputation_gs_%s.csv' % group_size_r
    f = open(file_name, 'w')

    start_time = datetime.datetime.now()

    run_time_r = 8
    sample_time_r = 2
    init_num = 2
    b_r_l = []
    for i in np.arange(1.0, 3.01, 0.1):
        b_r_l.append(round(i, 2))
    alpha_r = -2
    beta_r = -2
    result_r = []
    for defect_param_r in b_r_l:
        print('b value: ' + str(defect_param_r))
        round_result_r = []
        for _ in range(init_num):
            ind_strategy_r = initialize_strategy(total_num_r)
            group_rep_r = np.ones(group_num_r)
            for step_i in range(run_time_r):
                ind_strategy_r, group_rep_r = run_game(ind_strategy_r, alpha_r, beta_r, group_rep_r, defect_param_r,
                                                       group_size_r, group_base_r, group_length_r, total_num_r,
                                                       ind_pos_r, pos_ind_r)
            sample_strategy_r = []
            for step_i in range(sample_time_r):
                ind_strategy_r, group_rep_r = run_game(ind_strategy_r, alpha_r, beta_r, group_rep_r, defect_param_r,
                                                       group_size_r, group_base_r, group_length_r, total_num_r,
                                                       ind_pos_r, pos_ind_r)
                cal_strategy = np.zeros(2)
                for str_i in ind_strategy_r:
                    cal_strategy[str_i] += 1
                cal_strategy = cal_strategy / total_num_r
                sample_strategy_r.append(cal_strategy)
            round_result_r.append(np.mean(sample_strategy_r, axis=0))
        result_r.append(np.mean(round_result_r, axis=0))
    result_r_pd = pd.DataFrame(result_r, index=b_r_l)
    result_r_pd.to_csv(f)
    f.close()
    end_time = datetime.datetime.now()
    print(result_r)
    print(end_time - start_time)
