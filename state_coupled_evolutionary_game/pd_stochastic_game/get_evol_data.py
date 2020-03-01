import numpy as np
from evol_proc import evol_proc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b1', '--b1', type=float, default=2.0, help="The parameter of state 1")
parser.add_argument('-b2', '--b2', type=float, default=2.0, help="The parameter of state 2")
args = parser.parse_args()


def get_evol_data():
    """

    Returns:

    """
    # setting up the objects and defining the parameters
    beta = 1; b2 = args.b2; c = 1; b1 = args.b1; n_gen = 10**4; n_it = 100
    # Vectors that store the cooperation rates for each scenario in each round
    coops = np.zeros((1, n_gen)).flatten()
    coop1 = np.zeros((1, n_gen)).flatten()
    coop2 = np.zeros((1, n_gen)).flatten()
    # Vectors that store the average frequency of each memory-1 strategy
    freqs = np.zeros((1, 2 ** 2)).flatten()
    freq1 = np.zeros((1, 2 ** 2)).flatten()
    freq2 = np.zeros((1, 2 ** 2)).flatten()
    # Define the transitions of the three scenarios
    # In each q, there three cases, 0 C (DD), 1 C (CD or DC), 2C (CC)
    qs = np.array([[0.9, 0.9, 0.1], [0.1, 0.1, 0.9]]) # the scenario that transition between state 1 and state 2
    #q1 = np.array([[0, 0, 0], [1, 1, 1]]) # only in the state 1
    #q2 = np.array([[1, 1, 1], [0, 0, 0]]) # only in the state 2
    # Vector with all possible one-shot payoffs
    pi_round = np.array([0, b1, -c, b1 - c, 0, b2, -c, b2 - c])

    for i in range(n_it):  # run the evolution process with n_it initializations
        print(i)
        (coop, freq) = evol_proc(qs, pi_round, beta, n_gen)
        # print(coop.shape)
        # print(coop)
        # print(freq.shape)
        # print(freq)
        # get the average results_old of n_it initializations
        coops = i / (i + 1) * coops + 1 / (i + 1) * coop
        freqs = i / (i + 1) * freqs + 1 / (i + 1) * freq

        # (coop, freq) = evol_proc(q1, pi_round, beta, n_gen)
        # coop1 = i / (i + 1) * coop1 + 1 / (i + 1) * coop
        # freq1 = i / (i + 1) * freq1 + 1 / (i + 1) * freq
        #
        # (coop, freq) = evol_proc(q2, pi_round, beta, n_gen)
        # coop2 = i / (i + 1) * coop2 + 1 / (i + 1) * coop
        # freq2 = i / (i + 1) * freq2 + 1 / (i + 1) * freq

    coop = np.array([coops, coop1, coop2])
    freq = np.array([freqs, freq1, freq2])
    return (coop, freq)


def write_file(file_name, data_name):
    f = open(file_name, 'w')
    shape = data_name.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j < (shape[1] - 1):
                f.write(str(data_name[i][j]) + ',')
            else:
                f.write(str(data_name[i][j]))
        f.write('\n')
    f.close()


if __name__ == '__main__':
    coop_re, freq_re = get_evol_data()
    write_file('pri_coop_time.txt', coop_re)