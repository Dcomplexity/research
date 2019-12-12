import numpy as np
import networkx as nx
import pandas as pd
import math
import random
from itertools import permutations


def pd_game(a_x, a_y, r, s, t, p):
    if a_x == 1 and a_y == 1:
        p_x = r; p_y = r
    elif a_x == 1 and a_y == 0:
        p_x = s; p_y = t
    elif a_x == 0 and a_y == 1:
        p_x = t; p_y = s
    elif a_x == 0 and a_y == 0:
        p_x = p; p_y = p
    else:
        p_x = None; p_y = None
    return (p_x, p_y)


# Create the prisoners' dilemma game
def pd_game_b(a_x, a_y, b):
    if a_x == 1 and a_y == 1:
        p_x = 1; p_y = 1
    elif a_x == 1 and a_y == 0:
        p_x = 0; p_y = b
    elif a_x == 0 and a_y == 1:
        p_x = b; p_y = 0
    elif a_x == 0 and a_y == 0:
        p_x = 0; p_y = 0
    else:
        p_x = None; p_y = None
    return p_x, p_y


# Create donation game
def pd_donation_c_game(a_x, a_y, b, c):
    if a_x == 1 and a_y == 1:
        return b-c, b-c
    elif a_x == 1 and a_y == 0:
        return -c, b
    elif a_x == 0 and a_y == 1:
        return b, -c
    elif a_x == 0 and a_y == 0:
        return 0, 0
    else:
        return "Error"


def generate_well_mixed_network(popu_size):
    g_network = nx.complete_graph(popu_size)
    adj_array = nx.to_numpy_array(g_network)
    adj_link = []
    for i in range(adj_array.shape[0]):
        adj_link.append(np.where(adj_array[i] == 1)[0])
    g_edge = nx.Graph()
    for i in range(len(adj_link)):
        for j in range(len(adj_link[i])):
            g_edge.add_edge(i, adj_link[i][j])
    return np.array(adj_link), np.array(g_edge.edges())


def generate_lattice(popu_size, xdim, ydim):
    if popu_size != xdim * ydim:
        print("Wrong")
        return 0
    else:
        g_network = nx.grid_2d_graph(xdim, ydim, periodic=True)
        adj_array = nx.to_numpy_array(g_network)
        adj_link = []
        for i in range(adj_array.shape[0]):
            adj_link.append(np.where(adj_array[i] == 1)[0])
        g_edge = nx.Graph()
        for i in range(len(adj_link)):
            for j in range(len(adj_link[i])):
                g_edge.add_edge(i, adj_link[i][j])
        return np.array(adj_link), np.array(g_edge.edges())


# Generate the list of actions available in this game
def gen_actions():
    defect = 0
    cooperate = 1
    actions = [defect, cooperate]
    return actions


# Generate the list of states available in this game
def gen_states(actions):
    states = []
    for _ in permutations(actions, 2):
        states.append(_)
    for _ in actions:
        states.append((_, _))
    states.sort()
    return states


def alpha_time(time_step):
    return 1 / (10 + 0.0001 * time_step)


def epsilon_time(time_step):
    return 0.5 / (1 + 0.0001 * time_step)


class Agent:
    def __init__(self, agent_id, link, alpha=None, gamma=None, epsilon=None):
        self.agent_id = agent_id
        self.link = link
        self.payoff = 0
        self.time_step = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = gen_actions()
        self.len_a = len(self.actions)
        self.a_values = np.zeros(self.len_a)
        self.strategy = np.zeros(self.len_a)

    def get_id(self):
        return self.agent_id

    def get_link(self):
        return self.link[:]

    def get_actions(self):
        return self.actions

    def get_a_values(self):
        return self.a_values

    def get_strategy(self):
        return self.strategy

    def get_payoff(self):
        return self.payoff

    def set_payoff(self, n_payoff):
        self.payoff = n_payoff

    def add_payoff(self, n_payoff):
        self.payoff += n_payoff

    def set_time_step(self, t):
        self.time_step = t

    def set_strategy(self, other_strategy):
        self.strategy = other_strategy

    def set_alpha(self, t, new_alpha=None):
        if new_alpha:
            self.alpha = new_alpha
        else:
            self.alpha = alpha_time(t)

    def set_epsilon(self, t, new_epsilon=None):
        if new_epsilon:
            self.epsilon = new_epsilon
        else:
            self.epsilon = epsilon_time(t)

    def initial_strategy(self):
        """
        Initialize strategy, play each action by the same probability.
        :return:
        """
        for i in range(self.len_a):
            self.strategy[i] = 1 / self.len_a

    def initial_a_values(self):
        """
        Initialize the action values to all zeros
        :return:
        """
        for i in range(self.len_a):
            self.a_values[i] = 0

    def choose_action(self):
        pass

    def update_a_values(self, a):
        self.a_values[a] = (1 - self.alpha) * self.a_values[a] \
                           + self.alpha * (self.payoff + self.gamma * np.amax(self.a_values[:]))

    def update_strategy(self):
        pass

    def update_time_step(self):
        self.time_step += 1

    def update_alpha(self):
        self.alpha = 1 / (10 + 0.0001 * self.time_step)

    def update_epsilon(self):
        self.epsilon = 0.5 / (1 + 0.0001 * self.time_step)


class AgentImitation(Agent):
    def __init__(self, agent_id, link, alpha=None, gamma=None, epsilon=None, delta=None):
        Agent.__init__(self, agent_id, link, alpha, gamma, epsilon)
        self.delta = delta
        self.delta_table = np.zeros(self.len_a)
        self.delta_top_table = np.zeros(self.len_a)

    def choose_action(self):
        """
        Choose action epsilon-greedy
        :return:
        action: the chosen action
        :return:
        """
        if np.random.binomial(1, self.epsilon) == 1:
            a = np.random.choice(self.actions)
        else:
            a = np.random.choice(self.actions, size=1, p=self.strategy)[0]
        return a

    def imitation_strategy(self, p_j, a_i, a_j):
        for i in range(self.len_a):
            self.delta_table[i] = min(np.array([self.strategy[i], self.delta / (self.len_a - 1)]))
        sum_delta = 0
        if random.random() < 1 / (1 + math.e ** (2 * (self.payoff - p_j))):
            for act_i in [act_j for act_j in self.actions if act_j != a_j]:
                self.delta_top_table[act_i] = -self.delta_table[act_i]
                sum_delta += self.delta_table[act_i]
            self.delta_top_table[a_j] = sum_delta
            for i in range(self.len_a):
                self.strategy[i] += self.delta_top_table[i]
        else:
            for act_i in [act_j for act_j in self.actions if act_j != a_i]:
                self.delta_top_table[act_i] = -self.delta_table[act_i]
                sum_delta += self.delta_table[act_i]
            self.delta_top_table[a_i] = sum_delta
            for i in range(self.len_a):
                self.strategy[i] += self.delta_top_table[i]

    def valid_strategy(self):
        for i in range(self.len_a):
            if self.strategy[i] > 1.0:
                self.strategy[i] = 1.0
            if self.strategy[i] < 0.0:
                self.strategy[i] = 0.0


def initialize_population(popu_size, adj_link):
    popu = []
    for i in range(popu_size):
        popu.append(AgentImitation(i, adj_link[i], gamma=0.9, delta=0.0001))
    for i in range(popu_size):
        popu[i].initial_strategy()
        popu[i].initial_a_values()
        popu[i].set_time_step(t=0)
        popu[i].set_alpha(t=0)
        popu[i].set_epsilon(t=0)
    return popu


def imitation_process(popu, edge, r=3, s=0, t=5, p=1, b=1.0, c=1.0, b_c=1.0, game_type=None):
    total_num = len(popu)
    for i in range(total_num):
        popu[i].set_payoff(0)
    a_l = [0 for _ in range(total_num)]
    c_l = [0 for _ in range(total_num)]
    for i in range(total_num):
        a_l[i] = popu[i].choose_action()
    for pair in edge:
        ind_x = pair[0]
        ind_y = pair[1]
        if game_type == 'pd':
            p_x, p_y = pd_game(a_l[ind_x], a_l[ind_y], r, s, t, p)
        elif game_type == 'pd_b':
            p_x, p_y = pd_game_b(a_l[ind_x], a_l[ind_y], b)
        elif game_type == 'pd_donation_c':
            benifit = b_c * c
            p_x, p_y = pd_donation_c_game(a_l[ind_x], a_l[ind_y], c, benifit)
        else:
            p_x = 0; p_y = 0;
            print("wrong game type")
        popu[ind_x].add_payoff(p_x)
        popu[ind_y].add_payoff(p_y)
    for i in range(total_num):
        while True:
            ind_j = random.choice(range(total_num))
            if i != ind_j:
                break
        p_j = popu[ind_j].get_payoff()
        a_i = a_l[i]
        a_j = a_l[ind_j]
        popu[i].imitation_strategy(p_j, a_i, a_j)
        popu[i].valid_strategy()
        popu[i].update_time_step()
        popu[i].update_alpha()
        popu[i].update_epsilon()
    return popu


def run_imitation_process(popu_size, adj_link, edge, run_time, sample_time,
                          r=3, s=0, t=5, p=1, b=1.0, c=1.0, b_c=1.0, game_type=None):
    popu = initialize_population(popu_size, adj_link)
    for _ in range(run_time):
        popu = imitation_process(popu, edge, r, s, t, p, b, c, b_c, game_type)
    sample_strategy = []
    for _ in range(sample_time):
        popu = imitation_process(popu, edge, r, s, t, p, b, c, b_c, game_type)
        for i in range(popu_size):
            sample_strategy.append(popu[i].get_strategy())
    sample_strategy = np.mean(sample_strategy, axis=0)
    return sample_strategy

if __name__ == '__main__':
    popu_size = 100
    xdim = 10; ydim = 10
    run_time = 10000
    sample_time = 200
    r = 3; s = 0; t = 5; p = 1; b=0.8; c = 1.0; b_c = 2.4
    adj_link, edge = generate_well_mixed_network(popu_size)
    # adj_link, edge = generate_lattice(popu_size, xdim, ydim)
    # game_type = 'pd_donation_c'
    game_type = 'pd_b'
    result = []
    b_l = np.round(np.arange(0.0, 2.0, 0.1), 2)
    for b in b_l:
        print(b)
        one_result = run_imitation_process(popu_size, adj_link, edge, run_time, sample_time,
                          r, s, t, p, b, c, b_c, game_type)
        result.append(one_result)
    result_pd = pd.DataFrame(result, index=b_l)
    result_file = './results/pdd_well_mixed_imitation.csv'
    result_pd.to_csv(result_file)
    print(result_pd)




