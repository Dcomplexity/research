import datetime
import math
import os
import random

import pandas as pd

from game_env import *
from network_env import *


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
        return self.link[:]

    def get_strategy(self):
        return self.strategy

    def get_ostrategy(self):
        return self.ostrategy

    def get_action(self):
        if self.strategy == 0:
            return 0
        else:
            return 1

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
    network, total_num, edges = generate_network(xdim=30, ydim=30, structure='2d_grid')
    popu = []
    for i in range(total_num):
        # 0 for defection, 1 for cooperation, 2 for punishment
        popu.append(Agent(i, network[i], np.random.choice([0, 1, 2])))
    return popu, network, total_num, edges


def evolution_one_step(popu, total_num, edges, r):
    for i in range(total_num):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        a_l[i] = popu[i].get_action()
    for i in range(total_num):
        neigh = popu[i].get_link()
        neigh.append(i)
        enhancement_factor = r * len(neigh)
        a_neigh = []
        s_neigh = []
        for j in neigh:
            a_neigh.append(a_l[j])
            s_neigh.append(popu[j].get_strategy())
        p_neigh_l = play_pgg_game_punishment(a_neigh, enhancement_factor, s_neigh, 1, 0.5)
        for j in range(len(neigh)):
            popu[neigh[j]].add_payoffs(p_neigh_l[j])
    # Backup the strategy in this round
    for i in range(total_num):
        popu[i].set_ostrategy()
    # Update strategy by imitating other's strategy
    for i in range(total_num):
        ind = popu[i]
        w1 = 0.01
        w2 = random.random()
        if w2 < w1:
            potential_strategy = [0, 1, 2]
            potential_strategy.remove(ind.get_ostrategy())
            ind.set_strategy(np.random.choice(potential_strategy))
        else:
            ind_payoffs = ind.get_payoffs()
            j = random.choice(popu[i].get_link())
            opponent = popu[j]
            opponent_payoffs = opponent.get_payoffs()
            opponent_ostrategy = opponent.get_ostrategy()
            t1 = 1 / (1 + math.e ** (10.0 * (ind_payoffs - opponent_payoffs)))
            t2 = random.random()
            if t2 < t1:
                ind.set_strategy(opponent_ostrategy)
    return popu


def run(r, run_time):
    popu, network, total_num, edges = initialize_population()
    strategy_history = []
    for _ in range(run_time):
        strategy_dist = [0 for x in range(3)]
        for i in range(total_num):
            strategy_dist[popu[i].get_strategy()] += 1
        strategy_dist = np.array(strategy_dist) / total_num
        strategy_history.append(strategy_dist)
        popu = evolution_one_step(popu, total_num, edges, r)
    return strategy_history


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    simulation_name = "pgg_lattice_imitation_punishment"
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results_old/' + simulation_name + '/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


    run_time_r = 200
    r_r_l = []
    for i in np.arange(0.1, 2.01, 0.1):
        r_r_l.append(round(i, 2))
    for r_r in r_r_l:
        result_file_name = dir_name + 'strategy_history_%s.csv' % str(r_r)
        f = open(result_file_name, 'w')
        print('r value: ' + str(r_r))
        strategy_history_r = run(r_r, run_time_r)
        result_pd = pd.DataFrame(strategy_history_r, columns=['def_frac', 'co_frac', 'pun_frac'])
        result_pd.to_csv(f)
        f.close()
