import numpy as np
from collections import Counter

def play_pgg_game(a_l, r):
    agent_num = len(a_l)
    pf = np.sum(a_l) * r / agent_num
    pf_l = np.array([pf for _ in range(agent_num)]) - np.array(a_l)
    return pf_l

def play_pgg_game_discount_cost(a_l, r, discount):
    agent_num = len(a_l)
    pf = np.sum(a_l) * r / agent_num
    pf_l = np.array([pf for _ in range(agent_num)]) - np.array(a_l) * discount
    return pf_l


def play_pgg_game_punishment(a_l, r, s_l, p, p_cost):
    agent_num = len(a_l)
    pf = np.sum(a_l) * r / agent_num
    pf_l = np.array([pf for _ in range(agent_num)]) - np.array(a_l)
    s_counter = Counter(s_l)
    defector_num = s_counter[0]
    punisher_num = s_counter[2]
    defect_fine = punisher_num * p
    punishment_cost = defector_num * p_cost
    for i in range(len(s_l)):
        if s_l[i] == 0:
            pf_l[i] = pf_l[i] - defect_fine
        elif s_l[i] == 2:
            pf_l[i] = pf_l[i] - punishment_cost
    return pf_l




if __name__ == '__main__':
    pf_l_r = play_pgg_game([0, 1, 1], 2.0)
    print(pf_l_r)