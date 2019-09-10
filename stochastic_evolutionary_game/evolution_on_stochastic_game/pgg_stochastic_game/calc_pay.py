import numpy as np
import random
from scipy.linalg import null_space
import re

def calc_pay(strategy, qvec, r1, r2, c):
    """
    Calculates the payoff and cooperation rates in a stochastic game with
    deterministic transitions, playing a PGG in each state
    :param strategy: Matrix with n rows, each row contains the strategy of a player
    strategies have the form (pC,n-1, ...., pC,0, pD,n-1, ..., pD,0) where the letter
    refers to the player's previous own action and number refers to cooperators among co-players.
    :param qvec: [qn, ..., q0] vector that contains the transition
    probabilities qi to go to state 1 in the next round, depending on the number of cooperators.
    :param r1: multiplication factors of PGG in state 1
    :param r2: multiplication factors of PGG in state 2
    :param c: cost of cooperation
    :return:
    """
    # PART I --- Preparing a list of all possible states of the markov chain,
    # preparing a list of all possible payoffs in a given round

    # A state has the form (s, a1, ..., an) where s is the state of the
    # stochastic game and a1, ..., an are the player's actions.
    # Hence there are 2^(n+1) states.

    n = strategy.shape[0]
    # Matrix where each row corresponds to a possible state
    poss_state = np.zeros((2 ** (n + 1), n + 1))
    for i in range(2 ** (n + 1)):
        s = np.binary_repr(i, n + 1) # Return the binary representation of the input number as a string
        s_split = re.findall('\d', s)
        for j in range(n + 1):
            poss_state[i, j] = int(s_split[j])
    # Matrix where each row gives the payoff of all players in a given state
    pi_round = np.zeros((2 ** (n + 1), n))
    for i in range(2 ** (n + 1)):
        state = np.copy(poss_state[i])
        n_coop = np.sum(state[1:])
        mult = state[0] * r2 + (1 - state[0]) * r1 # state[0] = 0 for in state 1, state[0] = 1 for in state 2
        for j in range(n): # for every player
            pi_round[i, j] = n_coop * mult / n - state[j + 1] * c # state[j + 1] is the action of player j

    # PART II -- Creating the transition matrix between states
    m = np.zeros((2 ** (n + 1), 2 ** (n + 1)))
    ep = 0.001; strategy = (1 - ep) * strategy + ep * (1 - strategy)
    for row in range(2 ** (n + 1)):
        state_old = np.copy(poss_state[row]) # previous state
        n_coop = np.sum(state_old[1:])
        # qvec: [qn, ..., q0]: vector that contains the transition probabilities
        # qi: to go to state 1 in the next round, depending on the number of cooperators
        env_next = qvec[int(n - n_coop)]
        for col in range(2 ** (n + 1)):
            state_new = np.copy(poss_state[col]) # next state
            if state_new[0] == 1 - env_next:
                tr_pr = 1 # transition probability
                for i in range(n):
                    i_coop_old = state_old[1 + i]
                    pval = strategy[i, int(2 * n - 1 - n_coop - (n - 1) * i_coop_old)]
                    i_coop_next = state_new[1 + i]
                    tr_pr = tr_pr * (pval * i_coop_next + (1 - pval) * (1 - i_coop_next))
            else:
                tr_pr = 0
            m[row, col] = tr_pr

    # np.nan_to_num(m)
    null_matrix = np.transpose(m) - np.eye(2 ** (n + 1))
    # null_matrix.dropna(inplace = True)
    np.nan_to_num(null_matrix) # translate the nan and inf to a number (0 and 1.79769313e+308)
    v = null_space(null_matrix) # the shape of v is (2 ** (n + 1), 1)
    freq = np.transpose(v) / np.sum(v) # the shape of freq is (1, 2 ** (n + 1))
    # the shape of freq: (1, 2 ** (n + 1)), the shape of pi_round: (2 ** (n + 1), n)
    pivec = np.dot(freq, pi_round).flatten()
    cvec = np.sum(np.dot(freq, poss_state[:, 1:])) / n

    return (pivec, cvec)