import numpy as np
from evol_proc import evol_proc
import argparse
import pandas as pd
import os
parser = argparse.ArgumentParser()
parser.add_argument('-b1', '--b1', type=float, default=2.0, help="The parameter of state 1")
parser.add_argument('-b2', '--b2', type=float, default=2.0, help="The parameter of state 2")
parser.add_argument('-p1', '--p1', type=float, default=0.9, help="transition probability")
parser.add_argument('-p2', '--p2', type=float, default=0.1, help="transition probability")
args = parser.parse_args()


def get_evol_data():
    """

    Returns:

    """
    # setting up the objects and defining the parameters
    beta = 1; b2 = args.b2; c = 1; b1 = args.b1; n_gen = 10**4; n_it = 1000
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
    p1 = args.p1; p2 = args.p2
    qs = np.array([[p1, p1, p2], [p2, p2, p1]]) # the scenario that transition between state 1 and state 2
    #q1 = np.array([[0, 0, 0], [1, 1, 1]]) # only in the state 1
    #q2 = np.array([[1, 1, 1], [0, 0, 0]]) # only in the state 2
    # Vector with all possible one-shot payoffs
    pi_round = np.array([0, b1, -c, b1 - c, 0, b2, -c, b2 - c])

    freq_gen_it = []
    pop_count_gen_it = []
    pop_gen_it = []
    for i in range(n_it):  # run the evolution process with n_it initializations
        print(i)
        # (coop, freq) = evol_proc(qs, pi_round, beta, n_gen)
        freq_gen, pop_gen, pop_count = evol_proc(qs, pi_round, beta, n_gen)
        pop_count_gen_it.append(pop_count)
        freq_gen_it.append(freq_gen)
        pop_gen_it.append(pop_gen)
        # print(coop.shape)
        # print(coop)
        # print(freq.shape)
        # print(freq)
        # get the average results_old of n_it initializations
        # coops = i / (i + 1) * coops + 1 / (i + 1) * coop
        # freqs = i / (i + 1) * freqs + 1 / (i + 1) * freq

        # (coop, freq) = evol_proc(q1, pi_round, beta, n_gen)
        # coop1 = i / (i + 1) * coop1 + 1 / (i + 1) * coop
        # freq1 = i / (i + 1) * freq1 + 1 / (i + 1) * freq
        #
        # (coop, freq) = evol_proc(q2, pi_round, beta, n_gen)
        # coop2 = i / (i + 1) * coop2 + 1 / (i + 1) * coop
        # freq2 = i / (i + 1) * freq2 + 1 / (i + 1) * freq

    # coop = np.array([coops, coop1, coop2])
    # freq = np.array([freqs, freq1, freq2])
    return np.array(freq_gen_it), np.array(pop_gen_it), np.array(pop_count_gen_it)


# def write_file(file_name, data_name):
#     f = open(file_name, 'w')
#     shape = data_name.shape
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if j < (shape[1] - 1):
#                 f.write(str(data_name[i][j]) + ',')
#             else:
#                 f.write(str(data_name[i][j]))
#         f.write('\n')
#     f.close()


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.join(os.getcwd(), "./results"))
    freq_csv_file_name = "/p1_%.1f_p2_%.1f_b_%.1f_strategy_trace.csv" % (args.p1, args.p2, args.b1)
    pop_count_csv_file_name = "/p1_%.1f_p2_%.1f_b_%.1f_pop_count.csv" % (args.p1, args.p2, args.b1)
    pop_csv_file_name = "/p1_%.1f_p2_%.1f_b_%.1f_pop.csv" % (args.p1, args.p2, args.b1)
    freq_file_name = abs_path + freq_csv_file_name
    pop_count_file_name = abs_path + pop_count_csv_file_name
    pop_file_name = abs_path + pop_csv_file_name
    freq_gen_it, pop_gen_it, pop_count_gen_it = get_evol_data()
    freq_gen_it_mean = freq_gen_it.mean(axis=0)
    pop_count_gen_it_mean = pop_count_gen_it.mean(axis=0)
    pop_gen_it_mean = pop_gen_it.mean(axis=0)
    freq_gen_it_mean_pd = pd.DataFrame(freq_gen_it_mean)
    pop_count_gen_it_mean_pd = pd.DataFrame(pop_count_gen_it_mean)
    pop_gen_it_mean_pd = pd.DataFrame(pop_gen_it_mean)
    freq_gen_it_mean_pd.to_csv(freq_file_name, index=None)
    pop_count_gen_it_mean_pd.to_csv(pop_count_file_name, index=None)
    pop_gen_it_mean_pd.to_csv(pop_file_name, index=None)
