import numpy as np

from copy import deepcopy
from game_env import *

class Agent:
    def __init__(self, alpha, agent_id):
        self.id = agent_id
        self.strategy = []
        self.actions = [0, 1]  # 0 for defect and 1 for cooperate
        self.states = [0, 1]  # there are two states: state 0 and state 1
        self.alpha = alpha
        self.strategy_trace = []

    def get_strategy(self):
        return self.strategy

    def initial_strategy(self):
        for state in self.states:
            self.strategy.append(1 / len(self.actions))

    def set_strategy(self, new_s=None):
        """
        set a specified strategy
        :param new_s: the probability to play cooperation (action[1])
        :return:
        """
        if new_s:
            for state in self.states:
                self.strategy[state] = new_s[state]

    def choose_action(self, s):
        a = np.random.choice(np.array(self.actions), size=1, p=self.strategy[s])[0]
        return a

    def update_strategy(self, s, a, r):
        for action in self.actions:
            if action == a:
                self.strategy[s][action] = self.strategy[s][action] + self.alpha * r * (1 - self.strategy[s][action])
            else:
                self.strategy[s][action] = self.strategy[s][action] - self.alpha * r * self.strategy[s][action]

    def record_strategy(self):
        self.strategy_trace.append(deepcopy(self.strategy))


def run_game(agent_0_init_strategy, agent_1_init_strategy, s_0):
    agent_0 = Agent(alpha=0.001, agent_id=0)
    agent_1 = Agent(alpha=0.001, agent_id=1)
    agent_0.initial_strategy()
    agent_1.initial_strategy()
    agent_0.set_strategy(agent_0_init_strategy)
    agent_1.set_strategy(agent_1_init_strategy)
    cur_s = s_0
    games = [play_pd_game_1, play_pd_game_2]
    r_sum_0 = np.array([0.0, 0.0])  # store the sum of reward of agent_0 for each state: state 0 and state 1
    r_sum_1 = np.array([0.0, 0.0])
    tau_0 = np.array([0, 0])  # the reward of agent_0 used to update the strategy for each state
    tau_1 = np.array([0, 0])
    action_t_0 = np.array([0, 0])
    action_t_1 = np.array([0, 0])
    time_step = np.array([0, 0])  # store the sum of time of each state: state 0 and state 1
    visited = [0, 0]
    for _ in range(1000000):
        agent_0.record_strategy()
        agent_1.record_strategy()
        if visited[cur_s] == 0:
            visited[cur_s] = 1
            a_0 = agent_0.choose_action(cur_s)
            a_1 = agent_1.choose_action(cur_s)
            action_t_0[cur_s] = a_0
            action_t_1[cur_s] = a_1
            r_0, r_1 = games[cur_s](a_0, a_1)
            print(r_0, r_1)
            print(r_sum_0, r_sum_1)
            r_sum_0 = r_sum_0 + r_0
            r_sum_1 += r_1
            time_step += 1
            cur_s = next_state(cur_s, a_0, a_1)
        else:
            # In every time step, there is always one state will be visited
            # a_0 = agent_0.choose_action(cur_s)
            # a_1 = agent_1.choose_action(cur_s)
            tau_0[cur_s] = r_sum_0[cur_s] / time_step[cur_s]
            tau_1[cur_s] = r_sum_1[cur_s] / time_step[cur_s]
            # print(tau_0[cur_s], tau_1[cur_s])
            agent_0.update_strategy(cur_s, action_t_0[cur_s], tau_0[cur_s])
            agent_1.update_strategy(cur_s, action_t_1[cur_s], tau_1[cur_s])
            a_0 = agent_0.choose_action(cur_s)
            a_1 = agent_1.choose_action(cur_s)
            action_t_0[cur_s] = a_0
            action_t_1[cur_s] = a_1
            r_0, r_1 = games[cur_s](a_0, a_1)
            r_sum_0 += r_0
            r_sum_1 += r_1
            time_step[cur_s] = 0
            time_step += 1
            cur_s = next_state(cur_s, a_0, a_1)
    return agent_0.strategy_trace, agent_1.strategy_trace


if __name__ == '__main__':
    agent_0_strategy_trace, agent_1_strategy_trace = \
        run_game(agent_0_init_strategy={0: 0.3, 1: 0.3}, agent_1_init_strategy={0: 0.3, 1: 0.3}, s_0 = 0)
    print(agent_0_strategy_trace[-1], agent_1_strategy_trace[-1])
    agent_0_0_strategy = []

