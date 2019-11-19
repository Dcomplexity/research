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
    a_l = [0, 1]  # 0 for defect and 1 for cooperate
    a_num = len(a_l)
    network, total_num, edges = generate_network(xdim=10, ydim=10, structure='2d_grid')
    popu = []
    for i in range(total_num):
        popu.append(Agent(agent_id=i, action_num=a_num, link=network[i], gradient=True, gradient_base_line=True))
    return popu, network, total_num, edges


def rl_in_pgg(popu, total_num, edges, r, discount, eval=False):
    for i in range(total_num):
        popu[i].set_rewards(0)
    a_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        a_l[i] = popu[i].get_action()
    for i in range(total_num):
        neigh = popu[i].get_link()
        neigh.append(i)
        enhancement_factor = r * len(neigh)
        a_neigh = []
        for j in neigh:
            a_neigh.append(a_l[j])
        r_neigh_l = play_pgg_game_discount_cost(a_neigh, enhancement_factor, discount)
        for j in range(len(neigh)):
            popu[neigh[j]].add_rewards(r_neigh_l[j])

    for i in range(total_num):
        popu[i].take_action(a_l[i])

    if eval:
        co_frac = np.mean(a_l)
        return popu, co_frac
    else:
        return popu


def run(r, discount, run_time):
    popu, network, total_num, edges = initialize_population()
    for _ in range(run_time):
        popu = rl_in_pgg(popu, total_num, edges, r, discount, eval=False)
    return popu, network, total_num, edges


def evaluation(popu, edges, r, discount, sample_time):
    sample_co_frac = []
    total_num = len(popu)
    for _ in range(sample_time):
        popu, co_frac = rl_in_pgg(popu, total_num, edges, r, discount, eval=True)
        sample_co_frac.append(co_frac)
    return np.mean(sample_co_frac)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    discount_r = 0.5
    simulation_name = "pgg_lattice_rl_discount_cost_%s" % discount_r
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/results_old/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    result_file_name = dir_name + 'results_%s.csv' % simulation_name
    f = open(result_file_name, 'w')

    run_time_r = 5000
    sample_time_r = 200
    init_num = 5
    r_r_l = []
    for i in np.arange(0.1, 2.01, 0.1):
        r_r_l.append(round(i, 2))
    result_l = []
    for r_r in r_r_l:
        print('r value: ' + str(r_r))
        result = []
        for _ in range(init_num):
            popu_r, network_r, total_num_r, edges_r = run(r_r, discount_r, run_time_r)
            sample_result = evaluation(popu_r, edges_r, r_r, discount_r, sample_time_r)
            result.append(sample_result)
        result_l.append(np.mean(result))
    idx = pd.Index(r_r_l)
    idx.set_names('r')
    result_pd = pd.DataFrame(result_l, index=idx, columns=['co_frac'])
    # result_pd = pd.DataFrame({'r': r_r_l, 'co_frac': result_l})
    result_pd.to_csv(f)
    print(result_pd)
    f.close()





