import numpy as np
import pandas as pd
import random
import os
import math
import datetime
from game_env import *
from network_env import *


class Agent:
    def __init__(self, agent_id, action_num, link, epsilon=0, true_reward=0, initial=0,
                 step_size=0.01, sample_averages=False, UCB_param=None, gradient=False, gradient_base_line=False):
        self.agent_id = agent_id
        self.link = link
        self.rewards = 0
        self.time = 0
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.gradient = gradient
        self.UCB_param = UCB_param
        self.gradient_base_line = gradient_base_line
        self.average_reward = 0
        self.true_reward = true_reward

        self.a_num = action_num
        self.a_indices = np.arange(self.a_num)

        # real reward for each action
        self.q_true = []

        # estimation for each action
        self.q_est = np.zeros(self.a_num)

        # chosen times for each action
        self.action_count = []

        self.epsilon = epsilon

        for i in range(self.a_num):
            self.q_true.append(np.random.randn() + true_reward)
            self.q_est[i] = initial
            self.action_count.append(0)

        self.best_action = np.argmax(self.q_true)

    # get an action for this bandit, explore or exploit?
    def get_action(self):
        # explore
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                return np.random.choice(self.a_indices)

        # exploit
        if self.UCB_param is not None:
            UCB_est = self.q_est + \
                      self.UCB_param * np.sqrt(np.log(self.time + 1) / (np.asarray(self.action_count) + 1))
            return np.argmax(UCB_est)
        if self.gradient:
            exp_est = np.exp(self.q_est)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.a_indices, p=self.action_prob)
        return np.argmax(self.q_est)

    # take an action, update estimation for this action
    def take_action(self, action):
        # generate the reward under N (real reward, 1)
        # reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + self.rewards / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_est[action] += 1.0 / self.action_count[action] * (self.rewards - self.q_est[action])
        elif self.gradient:
            one_hot = np.zeros(self.a_num)
            one_hot[action] = 1
            if self.gradient_base_line:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_est = self.q_est + self.step_size * (self.rewards - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_est[action] += self.step_size * (self.rewards - self.q_est[action])

    def get_id(self):
        return self.agent_id

    def get_link(self):
        return self.link[:]
    #
    # def get_strategy(self):
    #     return self.strategy
    #
    # def get_ostrategy(self):
    #     return self.ostrategy
    #
    # def get_payoffs(self):
    #     return self.payoffs
    #
    # def set_strategy(self, other_strategy):
    #     self.strategy = other_strategy
    #
    # def set_ostrategy(self):
    #     self.ostrategy = self.strategy
    #
    def set_rewards(self, r):
        self.rewards = r

    def add_rewards(self, r):
        self.rewards = self.rewards + r


def initialize_population():
    a_l = [0, 1, 2]  # 0 for defect and 1 for cooperate
    a_num = len(a_l)
    network, total_num, edges = generate_network(xdim=10, ydim=10, structure='2d_grid')
    popu = []
    for i in range(total_num):
        popu.append(Agent(agent_id=i, action_num=a_num, link=network[i], gradient=True, gradient_base_line=True))
    return popu, network, total_num, edges


def rl_in_pgg(popu, total_num, edges, r, eval=False):
    for i in range(total_num):
        popu[i].set_rewards(0)
    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        a_l[i] = popu[i].get_action()

    group_num = 5

    for i in range(total_num):
        group = np.random.choice(np.arange(total_num), group_num, replace=False)
        enhancement_factor = r * group_num
        a_group = []
        for j in group:
            a_group.append(a_l[j])
        cd_group = []
        for a_group_item in a_group:
            if a_group_item == 0:
                cd_group.append(0)
            else:
                cd_group.append(1)
        r_group_l = play_pgg_game_punishment(cd_group, enhancement_factor, a_group, 1, 0.5)
        for j in range(len(group)):
            popu[group[j]].add_rewards(r_group_l[j])

    for i in range(total_num):
        popu[i].take_action(a_l[i])

    if eval:
        # co_frac = np.mean(a_l)
        return popu, a_l
    else:
        return popu


def run(r, run_time):
    popu, network, total_num, edges = initialize_population()
    a_l_history = []
    for _ in range(run_time):
        popu, a_l_record = rl_in_pgg(popu, total_num, edges, r, eval=True)
        a_l_dist = [0 for x in range(3)]
        for i in range(total_num):
            a_l_dist[a_l_record[i]] += 1
        a_l_dist = np.array(a_l_dist) / total_num
        a_l_history.append(a_l_dist)
    return a_l_history


if __name__ == '__main__':
    simulation_name = "pgg_well_mixed_rl_punishment"
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results_old/' + simulation_name + '/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    run_time_r = 5000
    r_r_l = []
    for i in np.arange(0.1, 2.01, 0.1):
        r_r_l.append(round(i, 2))
    for r_r in r_r_l:
        result_file_name = dir_name + "strategy_history_%s.csv" % str(r_r)
        f = open(result_file_name, 'w')
        print('r value: ' + str(r_r))
        action_history_r = run(r_r, run_time_r)
        result_pd = pd.DataFrame(action_history_r, columns=['def_frac', 'co_frac', 'pun_frac'])
        result_pd.to_csv(f)
        f.close()






