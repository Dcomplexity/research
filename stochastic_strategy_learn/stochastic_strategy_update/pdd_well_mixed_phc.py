import numpy as np
import networkx as nx
from itertools import permutations


def pdd_game(a_x, a_y, r, s, t, p):
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
    return 1 / (10 + 0.002 * time_step)


def epsilon_time(time_step):
    return 0.5 / (1 + 0.001 * time_step)


class Agent:
    def __init__(self, alpha=None, gamma=None, epsilon=None):
        self.time_step = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = gen_actions()
        self.states = gen_states(self.actions)
        self.q_table = {}
        self.strategy = {}

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_q_table(self):
        return self.q_table

    def get_strategy(self):
        return self.strategy

    def set_time_step(self, t):
        self.time_step = t

    def set_alpha(self, t, new_alpha=None):
        if new_alpha:
            self.alpha = new_alpha
        else:
            self.alpha = alpha_time(t)

    def set_epsilon(self, t, new_epsilon=None):
        if new_epsilon:
            self.epsilon = new_epsilon
        else:
            self.epsilon = 0.3

    def initial_strategy(self):
        """
        Initialize strategy, in each states, play each action by the same probability.
        :return:
        """
        len_actions = len(self.actions)
        initial_value = 1.0 / len_actions
        for i in self.states:
            self.strategy[i] = [0 for _ in range(len_actions)]
            for j in range(len_actions):
                self.strategy[i][j] = initial_value

    def initial_q_table(self):
        """
        Initialize the qTable to all zeros
        :return:
        """
        len_actions = len(self.actions)
        for i in self.states:
            self.q_table[i] = [0.0 for _ in range(len_actions)]

    def choose_action(self, ob):
        pass

    def update_q_table(self, s, a, r, s_):
        # Q-learning methods
        # self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * max(self.q_table[s_])
        self.q_table[s][a] += self.alpha * (q_target - q_predict)

    def update_strategy(self, s, a):
        pass

    def update_time_step(self):
        self.time_step += 1


class AgentFixedStrategy(Agent):
    def __init__(self, alpha=None, gamma=None, epsilon=None, fixed_strategy=None):
        Agent.__init__(self, alpha, gamma, epsilon)
        self.strategy_vector = fixed_strategy
        print(self.strategy_vector)

    def initial_strategy(self):
        len_actions = len(self.actions)
        for i in self.states:
            self.strategy[i] = [0 for _ in range(len_actions)]
        self.strategy[(1, 1)][0] = 1 - self.strategy_vector[0]
        self.strategy[(1, 1)][1] = self.strategy_vector[0]
        self.strategy[(1, 0)][0] = 1 - self.strategy_vector[1]
        self.strategy[(1, 0)][1] = self.strategy_vector[1]
        self.strategy[(0, 1)][0] = 1 - self.strategy_vector[2]
        self.strategy[(0, 1)][1] = self.strategy_vector[2]
        self.strategy[(0, 0)][0] = 1 - self.strategy_vector[3]
        self.strategy[(0, 0)][1] = self.strategy_vector[3]

    def choose_action(self, ob):
        a = np.random.choice(self.actions, size=1, p=self.strategy[ob])[0]
        return a

class AgentQ(Agent):
    def __init__(self, alpha=None, gamma=None, epsilon=None):
        Agent.__init__(self, alpha, gamma, epsilon)

    def choose_action(self, ob):
        a_v = np.array(self.q_table[ob])
        alt_actions = np.where(a_v == np.amax(a_v))[0]
        a = np.random.choice(alt_actions)
        return a


class AgentPHC(Agent):
    def __init__(self, alpha=None, gamma=None, epsilon=None, delta=None):
        Agent.__init__(self, alpha, gamma, epsilon)
        self.delta = delta
        self.delta_table = {}
        self.delta_top_table = {}

    def initial_delta(self):
        """
        Initialize the delta_table to all zeros.
        :return:
        """
        len_actions = len(self.actions)
        for i in self.states:
            self.delta_table[i] = np.zeros(len_actions)

    def choose_action(self, ob):
        """
        Choose action epsilon-greedy
        :param ob: the states agent's observation
        :return:
        action: the chosen action
        """
        if np.random.binomial(1, self.epsilon) == 1:
            a = np.random.choice(self.actions)
        else:
            a = np.random.choice(self.actions, size=1, p=self.strategy[ob])[0]
        return a

    def update_strategy(self, s, a):
        max_a = np.random.choice(np.argwhere(self.q_table[s] == np.amax(self.q_table[s])))[0]
        len_a = len(self.actions)
        for j in range(len_a):
            self.delta_table[s][j] = min(np.array[self.strategy[s][j], self.delta / (len_a - 1)])
        sum_delta = 0.0
        for act_i in [act_j for act_j in self.actions if act_j != max_a]:
            self.delta_top_table[s][act_i] = -self.delta_table[s][act_i]
            sum_delta += self.delta_table[s][act_i]
        self.delta_top_table[s][max_a] = sum_delta
        for j in range(len_a):
            self.strategy[s][j] += self.delta_top_table[s][j]

if __name__ == "__main__":
    A = Agent(0.1, 0.2, 0.3)
    A.initial_strategy()
    A.initial_q_table()
    q_table = A.get_q_table()
    strategy = A.get_strategy()
    print(q_table)
    print(q_table[(0, 0)][1])
