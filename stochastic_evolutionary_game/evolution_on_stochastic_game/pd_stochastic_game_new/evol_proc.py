import numpy as np
import random
from scipy.linalg import null_space


def evol_proc(qvec, pi_round, beta, n_gen):
    """
    Args:
        qvec: [[q10, q11, q12], [q20, q21, q22]]: transition probability to state 1,
        depending on previous number of cooperators, q10 means that in state 1, if there are 0 cooperators,
        the probability that next state is state 2.
        pi_round: [u1DD, u1DC, u1CD, u1CC, u2DD, u2DC, u2CD, u2CC]
        beta: selection strength
        n_gen: number of generations

    Returns:

    """
    # Setting up all objects
    n = 100
    # payoff vector from the perspective of player 1
    pv1 = np.copy(pi_round)
    # from the perspective of player 2
    pv2 = np.copy(pi_round); pv2[1:3] = pi_round[2:0:-1]; pv2[5:7] = pi_round[6:4:-1]
    # list of all strategy
    strategy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Initializing the pairwise payoff matrix and the cooperation matrix
    pay_m = np.zeros((4, 4)); c = np.zeros((4, 4))
    for i in range(4):
        for j in range(4)[i:4]:
            # Calculating and storing all pairwise payoffs and cooperation rates
            pi1, pi2, cop1, cop2 = payoff(strategy[i], strategy[j], qvec, pv1, pv2)
            pay_m[i, j] = pi1; pay_m[j, i] = pi2; c[i, j] = cop1; c[j, i] = cop2

    # Running the evolutionary process
    # Initialize all players use the strategy ALLD
    res = 0; pop = np.append(np.array([1]), np.zeros((3)))
    # Initialize the output vectors
    coop = np.zeros((1, n_gen)).flatten(); freq = np.zeros((1, 4)).flatten()
    for i in range(n_gen):
        # Introduce mutant strategy
        mut = random.choice(range(4))
        # Calculate fixation probability of mutant
        rho = calc_rho(mut, res, pay_m, n, beta)
        if random.random() < rho:
            res = mut
            pop = np.zeros((1, 4)).flatten()
            pop[res] = 1  # population state is updated
        coop[i] = c[res, res]  # Storing the cooperation rate at time i
        # It is equivalent to sum all pop and divide by n_gen
        print(freq, pop)
        freq = i / (i + 1) * freq + 1 / (i + 1) * pop # Updating the average frequency
    return (coop, freq)


def calc_rho(s1, s2, pay_m, n, beta):
    """
    Calculates the fixation probability of one s1 mutant in an s2 population
    Args:
        s1: mutant strategy s1
        s2: population strategy s2
        pay_m: payoffs matrix
        n: the number of players
        beta: selection strength

    Returns:
        rho: the probability of mutant strategy success
    """

    alpha = np.zeros((1, n-1)).flatten()
    for j in range(n)[1:]:  # j: number of mutants in the population
        # payoff of mutant
        pi1 = (j - 1) / (n - 1) * pay_m[s1, s1] + (n - j) / (n - 1) * pay_m[s1, s2]
        pi2 = j / (n - 1) * pay_m[s2, s1] + (n - j - 1) / (n - 1) * pay_m[s2, s2]
        alpha[j - 1] = np.exp(-beta * (pi1 - pi2))
    # Calculate the fixation probability according to formula given in SI
    # Indeed, the promotion of this method is reference 64 (Imitation process with small mutations) in SI section 2.3
    rho = 1 / (1 + np.sum(np.cumprod(alpha)))
    return rho


def payoff(p, q, qvec, piv1, piv2):
    """
    Calculate the payoff based on the strategy
    Args:
        p:
        q:
        qvec:
        piv1:
        piv2:

    Returns:

    """
    eps = 10 ** (-3) # Error rate for implementation errors
    p = p * (1 - eps) + (1 - p) * eps; q = q * (1 - eps) + (1 - q) * eps  # Adding errors to the players strategies
    M = np.array([[(1 - qvec[0][0]) * (1 - p[0]) * (1 - q[0]), (1 - qvec[0][0]) * (1 - p[0]) * q[0],
                   (1 - qvec[0][0]) * p[0] * (1 - q[0]), (1 - qvec[0][0]) * p[0] * q[0],
                   qvec[0][0] * (1 - p[0]) * (1 - q[0]), qvec[0][0] * (1 - p[0]) * q[0],
                   qvec[0][0] * p[0] * (1 - q[0]), qvec[0][0] * p[0] * q[0]],
                  [(1 - qvec[0][1]) * (1 - p[0]) * (1 - q[0]), (1 - qvec[0][1]) * (1 - p[0]) * q[0],
                   (1 - qvec[0][1]) * p[0] * (1 - q[0]), (1 - qvec[0][1]) * p[0] * q[0],
                   qvec[0][1] * (1 - p[0]) * (1 - q[0]), qvec[0][1] * (1 - p[0]) * q[0],
                   qvec[0][1] * p[0] * (1 - q[0]), qvec[0][1] * p[0] * q[0]],
                  [(1 - qvec[0][1]) * (1 - p[0]) * (1 - q[0]), (1 - qvec[0][1]) * (1 - p[0]) * q[0],
                   (1 - qvec[0][1]) * p[0] * (1 - q[0]), (1 - qvec[0][1]) * p[0] * q[0],
                   qvec[0][1] * (1 - p[0]) * (1 - q[0]), qvec[0][1] * (1 - p[0]) * q[0],
                   qvec[0][1] * p[0] * (1 - q[0]), qvec[0][1] * p[0] * q[0]],
                  [(1 - qvec[0][2]) * (1 - p[0]) * (1 - q[0]), (1 - qvec[0][2]) * (1 - p[0]) * q[0],
                   (1 - qvec[0][2]) * p[0] * (1 - q[0]), (1 - qvec[0][2]) * p[0] * q[0],
                   qvec[0][2] * (1 - p[0]) * (1 - q[0]), qvec[0][2] * (1 - p[0]) * q[0],
                   qvec[0][2] * p[0] * (1 - q[0]), qvec[0][2] * p[0] * q[0]],
                  [qvec[1][0] * (1 - p[1]) * (1 - q[1]), qvec[1][0] * (1 - p[1]) * q[1],
                   qvec[1][0] * p[1] * (1 - q[1]), qvec[1][0] * p[1] * q[1],
                   (1 - qvec[1][0]) * (1 - p[1]) * (1 - q[1]), (1 - qvec[1][0]) * (1 - p[1]) * q[1],
                   (1 - qvec[1][0]) * p[1] * (1 - q[1]), (1 - qvec[1][0]) * p[1] * q[1]],
                  [qvec[1][1] * (1 - p[1]) * (1 - q[1]), qvec[1][1] * (1 - p[1]) * q[1],
                   qvec[1][1] * p[1] * (1 - q[1]), qvec[1][1] * p[1] * q[1],
                   (1 - qvec[1][1]) * (1 - p[1]) * (1 - q[1]), (1 - qvec[1][1]) * (1 - p[1]) * q[1],
                   (1 - qvec[1][1]) * p[1] * (1 - q[1]), (1 - qvec[1][1]) * p[1] * q[1]],
                  [qvec[1][1] * (1 - p[1]) * (1 - q[1]), qvec[1][1] * (1 - p[1]) * q[1],
                   qvec[1][1] * p[1] * (1 - q[1]), qvec[1][1] * p[1] * q[1],
                   (1 - qvec[1][1]) * (1 - p[1]) * (1 - q[1]), (1 - qvec[1][1]) * (1 - p[1]) * q[1],
                   (1 - qvec[1][1]) * p[1] * (1 - q[1]), (1 - qvec[1][1]) * p[1] * q[1]],
                  [qvec[1][2] * (1 - p[1]) * (1 - q[1]), qvec[1][2] * (1 - p[1]) * q[1],
                   qvec[1][2] * p[1] * (1 - q[1]), qvec[1][2] * p[1] * q[1],
                   (1 - qvec[1][2]) * (1 - p[1]) * (1 - q[1]), (1 - qvec[1][2]) * (1 - p[1]) * q[1],
                   (1 - qvec[1][2]) * p[1] * (1 - q[1]), (1 - qvec[1][2]) * p[1] * q[1]],
                  ])
    null_matrix = np.transpose(M) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    piv1 = piv1.reshape(piv1.size, 1).transpose()  # shape is (1, 8)
    piv2 = piv2.reshape(piv2.size, 1).transpose()
    pi1 = np.dot(piv1, v)[0]
    pi2 = np.dot(piv2, v)[0]
    v = v.flatten()
    cop1 = v[2] + v[3] + v[6] + v[7]
    cop2 = v[1] + v[3] + v[5] + v[7]
    return (pi1, pi2, cop1, cop2)