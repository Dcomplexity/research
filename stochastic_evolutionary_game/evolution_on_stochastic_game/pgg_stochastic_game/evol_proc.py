import numpy as np
import random
from scipy.special import comb
import re
from numpy.matlib import repmat

from calc_pay import calc_pay

def evol_proc(qvec, r1, r2, c, beta, n_gen):
    """
    The process of evolution
    :param qvec: qvec = [qn ... q0], transition probability to state 1, depending on previous number of cooperators
    :param r1: multiplication factors in state 1
    :param r2: multiplication factors in state 2
    :param c: cost of cooperation
    :param beta: strength of selection
    :param n_gen: number of mutants considered
    :return:
    coop: average cooperation rate
    freq: average abundance for each memory-1 strategy
    """

    # Setting up all objects
    pop_size = 100 # population size
    n =  qvec.shape[0] - 1 # number of players
    global binom, all_strategy
    # pre-calculating all possible binomial coefficients that will be needed
    binom = calc_binom(pop_size, n)
    # strategies have the form(pC,n-1, ...., pC,0, pD,n-1, ..., pD,0)
    # Construct a list of all possible strategies
    # Each item is 0 or 1, or there are total 2**(2*n) strategies
    all_strategy = np.zeros((2 ** (2 * n), 2 * n))
    ns = 2 ** (2 * n) # number of strategies
    for i in range(ns):
        s = np.binary_repr(i, 2 * n)
        s_split = re.findall('\d', s)
        for j in range(2 * n):
            all_strategy[i, j] = int(s_split[j])

    # Initialize a vector that contains all payoffs and cooperation rates in homogeneous population
    pay_h = np.zeros((1, ns)).flatten()
    coop_h = np.zeros((1, ns)).flatten()
    for i in range(ns): # Calculate the values of pay_h and coop_h
        strategy_h = np.zeros((n, 2 * n))
        for j in range(n):
            strategy_h[j] = np.copy(all_strategy[i])
        (pivec, cvec) = calc_pay(strategy_h, qvec, r1, r2, c)
        pay_h[i] = pivec[0]
        coop_h[i] = cvec

    # Run the evolutionary process
    # Initialize population: All D
    res = 0
    pop = np.zeros((1, 2 ** (2 * n))).flatten()
    pop[res] = 1
    # Initialize the output
    coop = np.zeros((1, n_gen)).flatten()
    freq = np.zeros((1, 2 ** (2 * n))).flatten()
    for i in range(n_gen):
        # introduce a mutant strategy
        mut = random.choice(range(2 ** (2 * n)))
        # Calculate fixation probability of mutant
        rho = calc_rho(mut, res, pay_h, pop_size, n, qvec, r1, r2, c, beta)
        if random.random() < rho: # if fixation occurs
            res = mut # resident strategy is replaced by mutant strategy
            # population state is updated
            pop = np.zeros((1, 2 ** (2 * n))).flatten()
            pop[res] = 1
        coop[i] = coop_h[res] # store the cooperation rate at time i
        freq = i / (i + 1) * freq + 1 / (i + 1) * pop

    return (coop, freq)


def calc_rho(s1, s2, pay_h, pop_size, n, rv, r1, r2, c, beta):
    """
    Calculates the fixation probability of one s1 mutant in an s2 population
    :param s1: strategy s1
    :param s2: strategy s2
    :param pay_h: matrix of payoff
    :param pop_size: population size
    :param n: group size
    :param rv: transition probability between states
    :param r1: multiplication of state 1
    :param r2: multiplication of state 2
    :param c: cost of public goods game
    :param beta: strength of mutation
    :return:
    """
    alpha = np.zeros((1, pop_size - 1)).flatten()

    # first step: calculate the payoff of an s1 player and an s2 player, depending on number of s1 players in the group
    pay = np.zeros((n + 1, 2)) # matrix that contains the payoffs of the two players
    pay[n, 0] = pay_h[s1] # entry (n, 0) ... everyone plays s1
    pay[0, 1] = pay_h[s2] # entry (1, 2) ... everyone plays s2
    strategy1 = all_strategy[s1]; strategy2 = all_strategy[s2] # two strategies
    for n_mut in range(n - 1): # number of mutants
        strategy = np.append(repmat(strategy1, n_mut + 1, 1), repmat(strategy2, n - n_mut - 1, 1), axis=0)
        # calculate and store payoffs for that group
        (pi_vec, c_vec) = calc_pay(strategy, rv, r1, r2, c)
        pay[n_mut + 1, 0] = pi_vec[0]; pay[n_mut + 1, 1] = pi_vec[-1]
    for j in range(pop_size - 1): # j + 1 corresponds to the number of s1 players in the whole population
        # if j = 0, means there is 1 s1 player, the s1 player will face the group that contains all s2 players
        # pay[1:, 0] is for the payoff that there are at least one s1 player
        # from the perspective of s2, if there is one s1 player, then the probabilities of possible groups
        # are stored in binom[j + 1]
        # Consider all possible groups the s1 player could find herself in
        pi1 = np.dot(binom[j], pay[1:, 0].reshape(pay.shape[0] - 1, 1))
        # Consider all possible groups the s2 player could find herself in
        pi2 = np.dot(binom[j + 1], pay[0: -1, 1].reshape(pay.shape[0] - 1, 1))
        alpha[j] = np.exp(-beta * (pi1[0] - pi2[0]))
    rho = 1 / (1 + np.sum(np.cumprod(alpha)))
    return rho


def calc_binom(pop_size, n):
    """
    Calculates the probability of a certain group composition given there are only two strategies present.
    row - 1 ... number of players with strategy 1 among all other players in population
    col - 1 ... number of co-players with strategy 1 in group
    :param pop_size: the total number
    :param n: the number of pickup things
    :return:
    bm: a pop_size * n matrix
    """
    bm = np.zeros((pop_size, n))
    for row in range(pop_size):
        for col in range(n):
            bm[row, col] = comb(row, col) * comb(pop_size - row - 1, n - col - 1) / comb(pop_size - 1, n - 1)
    return bm