import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd


class Agent:
    def __init__(self, agent_id, a):
        self.agent_id = agent_id
        self.a = a
        self.old_a = a
        self.p = 0

    def get_id(self):
        return self.agent_id

    def get_action(self):
        return self.a

    def get_old_action(self):
        return self.old_a

    def get_payoffs(self):
        return self.p

    def set_action(self, other_a):
        self.a = other_a

    def set_old_action(self):
        self.old_a = self.a

    def set_payoffs(self, new_p):
        self.p = new_p

    def add_payoffs(self, added_p):
        self.p = self.p + added_p


def init(popu_s, k):
    """
    Initialize the population
    :param popu_s: the size of population
    :param k: the initial num of cooperators
    :return:
    a list of agents
    """
    popu = []
    popu_index = list(range(popu_s))
    random.shuffle(popu_index)
    s = 0
    for i in popu_index:
        if s < k: # initial action is cooperative action
            popu.append(Agent(i, 1))
        else:
            popu.append(Agent(i, 0))
        s = s + 1
    return popu


# 0 for defection and 1 for cooperation
def play_game(a_x, a_y, r, s, t, p):
    if a_x == 1 and a_y == 1:
        return r, r
    elif a_x == 1 and a_y == 0:
        return s, t
    elif a_x == 0 and a_y == 1:
        return t, s
    elif a_x == 0 and a_y == 0:
        return p, p
    else:
        return "Error"


def evolution_one_step(popu, beta, r, s, t, p):
    popu_s = len(popu)
    for i in range(popu_s):
        popu[i].set_payoffs(0)
    a_l = [0 for _ in range(popu_s)]
    for i in range(popu_s):
        a_l[i] = popu[i].get_action()
    pairs = list(itertools.combinations(range(popu_s), 2))
    for agent_pair in pairs:
        i = agent_pair[0]
        j = agent_pair[1]
        r_i, r_j = play_game(a_l[i], a_l[j], r, s, t, p)
        popu[i].add_payoffs(r_i)
        popu[j].add_payoffs(r_j)
    # Backup the action in this round
    for i in range(popu_s):
        popu[i].set_old_action()
    # Update action by imitating other's action
    learner = random.choice(range(popu_s))
    while True:
        role = random.choice(range(popu_s))
        if learner != role:
            break
    t1 = 1 / (1 + math.e ** (-1 * beta * (popu[role].get_payoffs() - popu[learner].get_payoffs())))
    t2 = random.random()
    if t2 < t1:
        popu[learner].set_action(popu[role].get_old_action())
    return popu


def run(popu_s, k, beta, r, s, t, p, r_time):
    popu = init(popu_s, k)
    for _ in range(r_time):
        popu = evolution_one_step(popu, beta, r, s, t, p)
        coop_num = 0
        for i in range(popu_s):
            coop_num += popu[i].get_action()
        if coop_num == popu_s or coop_num == 0:
            break
    return popu


def get_fixation_probability(popu_s, k, beta, r, s, t, p, r_time, init_time):
    f_p = 0
    for i in range(init_time):
        popu = run(popu_s, k, beta, r, s, t, p, r_time)
        coop_num = 0
        for _ in range(popu_s):
            coop_num += popu[_].get_action()
        coop_frac = coop_num / popu_s
        f_p = i / (i + 1) * f_p + 1 / (i + 1) * coop_frac
    return f_p


if __name__ == '__main__':
    popu_s = 20
    beta = 0.1
    r = 3; s = 1; t = 4; p = 2
    init_time = 2000
    run_time = 10000
    start_time = datetime.datetime.now()
    result = []
    for k in range(21):
        print(k)
        f_p = get_fixation_probability(popu_s, k, beta, r, s, t, p, run_time, init_time)
        result.append(f_p)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    result = pd.DataFrame({'phi_k': result})
    file_name = './results/%.3f_simulation_result.csv' % beta
    result.to_csv(file_name)
    print(result)

