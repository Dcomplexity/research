from game_env import *
from network_env import *
import numpy as np
import pandas as pd
import math
import random
import os
import datetime


class Agent:
    def __init__(self, agent_id, link, strategy):
        self.agent_id = agent_id
        self.link = link
        self.strategy = strategy
        self.ostrategy = strategy
        self.payoffs = 0

    def get_id(self):
        return self.agent_id

    def get_link(self):
        return self.link

    def get_strategy(self):
        return self.strategy

    def get_ostrategy(self):
        return self.ostrategy

    def get_payoffs(self):
        return self.payoffs

    def set_strategy(self, other_strategy):
        self.strategy = other_strategy

    def set_ostrategy(self):
        self.ostrategy = self.strategy

    def set_payoffs(self, p):
        self.payoffs = p

    def add_payoffs(self, p):
        self.payoffs = self.payoffs + p


def initialize_population():
    network, total_num, edges = generate_network(structure='2d_grid')
    popu = []
    for i in range(total_num):
        popu.append(Agent(i, network[i], np.random.choice([0, 1, 2, 3])))
    return popu, network, total_num, edges


def evolution_one_step(popu, total_num, edges, b):
    for i in range(total_num):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        if popu[i].get_strategy() == 0:
            a_l[i] = 0
        else:
            a_l[i] = 1
    for co_pair in edges:
        co_i = co_pair[0]
        co_j = co_pair[1]
        r_i, r_j = play_b_game(a_l[co_i], a_l[co_j], b)
        popu[co_i].add_payoffs(r_i)
        popu[co_j].add_payoffs(r_j)
    # for i in range(total_num):
    #     play_agent_l = popu[i].get_link()
    #     for j in play_agent_l:
    #         r_i, r_j = play_donation_game(a_l[i], a_l[j], b)
    #         popu[i].add_payoffs(r_i)
    #         popu[j].add_payoffs(r_j)
    # for i in range(total_num):
    #     if popu[i].get_strategy() == 3:
    #         play_agent_l = popu[i].get_link()
    #         for j in play_agent_l:
    #             if a_l[j] == 0:
    #                 popu[i].add_payoffs(-b)
    #                 popu[j].add_payoffs(-b)
    a_new_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        if popu[i].get_strategy() == 0 or popu[i].get_strategy() == 1:
            a_new_l[i] = 0
        else:
            a_new_l[i] = 1
    for co_pair in edges:
        co_i = co_pair[0]
        co_j = co_pair[1]
        if a_l[co_i] == 1 and a_l[co_j] == 1:
            r_i, r_j = play_b_game(a_new_l[co_i], a_new_l[co_j], b)
            popu[co_i].add_payoffs(r_i)
            popu[co_j].add_payoffs(r_j)
    for i in range(total_num):
        if popu[i].get_strategy() == 3:
            play_agent_l = popu[i].get_link()
            for j in play_agent_l:
                if popu[j].get_strategy() == 1:
                    popu[i].add_payoffs(-b)
                    popu[j].add_payoffs(-b)
    # for i in range(total_num):
    #     if a_l[i] == 1:
    #         neigh_agent = popu[i].get_link()
    #         poten_agent = []
    #         for j in neigh_agent:
    #             if a_l[j] == 1:
    #                 poten_agent.append(j)
    #         for j in poten_agent:
    #             r_i, r_j = play_donation_game(a_new_l[i], a_new_l[j], b)
    #             popu[i].add_payoffs(r_i)
    #             popu[j].add_payoffs(r_j)
    #         if popu[i].get_strategy() == 3:
    #             for j in poten_agent:
    #                 if a_new_l[j] == 0:
    #                     popu[i].add_payoffs(-(b-1.0))
    #                     popu[j].add_payoffs(-(b-1.0))
    # Backup the strategy in this round
    for i in range(total_num):
        popu[i].set_ostrategy()
    # Update strategy by imitating other's strategy
    for i in range(total_num):
        ind = popu[i]
        w1 = 0.01 #  mutation probability
        w2 = random.random()
        if w2 < w1:
            potential_strategy = [0, 1, 2, 3]
            potential_strategy.remove(ind.get_ostrategy())
            ind.set_strategy(np.random.choice(potential_strategy))
        else:
            ind_payoffs = ind.get_payoffs()
            j = random.choice(popu[i].get_link())
            opponent = popu[j]
            opponent_payoffs = opponent.get_payoffs()
            opponent_ostrategy = opponent.get_ostrategy()
            t1 = 1 / (1 + math.e ** (2.0 * (ind_payoffs - opponent_payoffs)))
            t2 = random.random()
            if t2 < t1:
                ind.set_strategy(opponent_ostrategy)
    return  popu


def run(b, run_time):
    run_time = run_time
    popu, network, total_num, edges = initialize_population()
    for _ in range(run_time):
        popu = evolution_one_step(popu, total_num, edges, b)
    return popu, network, total_num, edges


def evaluation(popu, edges, b, sample_time):
    sample_time = sample_time
    sample_strategy = []
    total_num = len(popu)
    for _ in range(sample_time):
        popu = evolution_one_step(popu, total_num, edges, b)
        strategy_dist = [0 for _ in range(4)]
        for i in range(total_num):
            strategy_dist[popu[i].get_strategy()] += 1
        strategy_dist = np.array(strategy_dist) / total_num
        sample_strategy.append(strategy_dist)
    return sample_strategy


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    simulation_name = "b_lattice_reward_cheat_punishment"
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    result_file_name = dir_name + "results_%s.csv" % simulation_name
    f = open(result_file_name, 'w')

    run_time_r = 80
    sample_time_r = 20
    init_num = 5
    b_r_l = []
    for i in np.arange(1.0, 3.01, 0.1):
        b_r_l.append(round(i, 2))
    result_l = []
    for b_r in b_r_l:
        print('b value: ' + str(b_r))
        result = []
        for _ in range(init_num):
            popu_r, network_r, total_num_r, edges_r = run(b_r, run_time_r)
            sample_result = evaluation(popu_r, edges_r, b_r, sample_time_r)
            for sample_i in sample_result:
                result_l.append(sample_i)
    init_num_l = list(range(init_num))
    sample_l = list(range(sample_time_r))
    m_index = pd.MultiIndex.from_product([b_r_l, init_num_l, sample_l], names=['b', 'init', 'sample'])
    result_pd = pd.DataFrame(result_l, index=m_index)
    result_pd.to_csv(f)
    f.close()
    end_time = datetime.datetime.now()
    print(end_time - start_time)