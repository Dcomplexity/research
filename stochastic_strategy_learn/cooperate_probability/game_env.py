import numpy as np


def play_pgg_game(a_l, r):
    agent_num = len(a_l)
    pf = np.sum(a_l) * r / agent_num
    pf_l = np.array([pf for _ in range(agent_num)]) - np.array(a_l) / 10
    return pf_l


if __name__ == '__main__':
    pf_l_r = play_pgg_game([0, 1, 1], 2.0)
    print(pf_l_r)