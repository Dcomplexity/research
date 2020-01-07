import numpy as np
import random
import matplotlib.pyplot as plt


# In this work, we define 0 as cooperation and 1 as defection
def play_game(a_x, a_y, r, s, t, p):
    if a_x == 0 and a_y == 0:
        return r, r
    elif a_x == 0 and a_y == 1:
        return s, t
    elif a_x == 1 and a_y == 0:
        return t, s
    elif a_x == 1 and a_y == 1:
        return p, p
    else:
        return "Error"


def choose_action(s):
    if np.random.random() < s:
        return 0
    else:
        return 1


def reactive_learning(s_p, s_q, r, s, t, p):
    a_p = 0
    a_q = 0
    delta_p = 0.0
    p_list = []
    for i in range(1000):
        if i == 0:
            a_p = choose_action(s_p[0])
            a_q = choose_action(s_q[0])
            payoff_p, payoff_q = play_game(a_p, a_q, r, s, t, p)
            delta_p = s_p[0]
            p_list.append([payoff_p, payoff_q])
        else:
            a_p_last = a_p
            a_q_last = a_q
            if a_q_last == 0:
                delta_p = delta_p * s_p[1] + (1 - delta_p) * s_p[3]
            else:
                delta_p = delta_p * s_p[2] + (1 - delta_p) * s_p[4]
            a_p = choose_action(delta_p)
            a_q = choose_action(s_q[a_q_last * 2 + a_p_last + 1])
            payoff_p, payoff_q = play_game(a_p, a_q, r, s, t, p)
            p_list.append([payoff_p, payoff_q])
    return np.array(p_list).mean(axis=0)


if __name__ == '__main__':
    # s_p_r = [0.50, 11 / 13, 1 / 2, 7 / 26, 0]
    s_p_r = [0.50, 0.5, 0.5, 0.5, 0.5]
    # s_p_r = [0.50, 0.99, 0.40, 0.01, 0.01]
    # s_q_r = [0.25, 0.25, 0.25, 0.25, 0.25]
    r_r = 3
    s_r = 0
    t_r = 5
    p_r = 1
    payoff_pair = []
    for _ in np.arange(10e3):
        if _ % 10e3 == 0:
            print(_)
        s_q_r = np.random.beta(0.5, 0.5, 5)
        payoff_pair.append(reactive_learning(s_p_r, s_q_r, r_r, s_r, t_r, p_r))
    # print(payoff_pair)
    payoff_pair = np.array(payoff_pair)
    plt.xlim(left = 0, right = 5)
    plt.ylim(bottom = 0, top = 5)
    plt.scatter(payoff_pair[:, 1], payoff_pair[:, 0], s=2.0, c='green')
    plt.savefig('./images/reactive_learn_strategy.png')
    plt.show()