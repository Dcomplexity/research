import numpy as np
from evol_proc import evol_proc
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r1", "--r1", type=float, default=1.6, help="The parameter of state 1")
parser.add_argument("-r2", "--r2", type=float, default=1.2, help="The parameter of state 2")
args = parser.parse_args()

def get_evol_data():
    """
    Creates the data for the dynamics of cooperation rate with time
    Simulates the dynamics of the stochastic game, and of the two corresponding repeated game
    :return:
    coop: matrix that contains the average cooperation rate in population
    for each time step and each of the three scenarios
    freq: matrix that contains the average abundance of each strategy
    data: stores the parameters used for the simulation
    """

    # Set up the objects and define the parameters (Parameters in fig. 2b)
    gr_size = 4; beta = 100; r2 = 1.2; r1 = 1.6; c = 1; n_gen = 10 ** 4; n_int = 100
    coop_s = np.zeros((1, n_gen)).flatten()
    coop_1 = np.zeros((1, n_gen)).flatten()
    coop_2 = np.zeros((1, n_gen)).flatten()
    freq_s = np.zeros((1, 2 ** (2 * gr_size))).flatten()
    freq_1 = np.zeros((1, 2 ** (2 * gr_size))).flatten()
    freq_2 = np.zeros((1, 2 ** (2 * gr_size))).flatten()
    # Define the transitions of the three scenarios
    q_s = np.append(np.array([1]), np.zeros((gr_size)))
    q_1 = np.ones((1, gr_size + 1)).flatten()
    q_2 = np.zeros((1, gr_size + 1)).flatten()

    for i in range(n_int):
        print(i)
        start_time = datetime.datetime.now()
        (coop, freq) = evol_proc(q_s, r1, r2, c, beta, n_gen)
        coop_s = i / (i + 1) * coop_s + 1 / (i + 1) * coop
        freq_s = i / (i + 1) * freq_s + 1 / (i + 1) * freq
        end_time_s = datetime.datetime.now()
        print("complete state s", (end_time_s - start_time).seconds)
        (coop, freq) = evol_proc(q_1, r1, r2, c, beta, n_gen)
        coop_1 = i / (i + 1) * coop_1 + 1 / (i + 1) * coop
        freq_1 = i / (i + 1) * freq_1 + 1 / (i + 1) * freq
        end_time_1 = datetime.datetime.now()
        print("complete state 1", (end_time_1 - start_time).seconds)
        (coop, freq) = evol_proc(q_2, r1, r2, c, beta, n_gen)
        coop_2 = i / (i + 1) * coop_2 + 1 / (i + 1) * coop
        freq_2 = i / (i + 1) * freq_2 + 1 / (i + 1) * freq
        end_time_2 = datetime.datetime.now()
        print("complete state 2", (end_time_2 - start_time).seconds)

    # Create the output
    coop = np.array([coop_s, coop_1, coop_2])
    freq = np.array([freq_s, freq_1, freq_2])
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
    (coop_re, fre_re) = get_evol_data()
    write_file('pub_coop_time.txt', coop_re)