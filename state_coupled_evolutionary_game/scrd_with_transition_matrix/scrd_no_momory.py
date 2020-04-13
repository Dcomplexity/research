import os
import pandas as pd
import datetime
import numpy as np
from sympy import Matrix
from scipy.linalg import null_space

def valid_s(s_value):
    if s_value < 0.001:
        s_new = 0.001
    elif s_value > 0.999:
        s_new = 0.999
    else:
        s_new = s_value
    return s_new


def build_markov_chain(qvec, p, q):
    # m = np.array([[(-1 + p[0] * q[0]), (-1 + p[0]), (-1 + q[0]), f[0]],
    #               [p[1] * q[2], (-1 + p[1]), q[2], f[1]],
    #               [p[2] * q[1], p[2], (-1 + q[1]), f[2]],
    #               [p[3] * q[3], p[3], q[3], f[3]]])
    # m = np.array([[f[0], (-1 + p[0]), (-1 + q[0]), (1-p[0])*(1-q[0])],
    #               [f[1] * q[2], (-1 + p[1]), q[2], (1-p[1])*(1-q[1])],
    #               [f[2] * q[1], p[2], (-1 + q[1]), (1-p[2])*(1-q[2])],
    #               [f[3] * q[3], p[3], q[3], (1-p[3])*(1-q[3])-1]])

    m = np.array([[qvec[0] * p[0] * q[0], qvec[0] * p[0] * (1 - q[0]), qvec[0] * (1 - p[0]) * q[0],
                   qvec[0] * (1 - p[0]) * (1 - q[0]),
                   (1 - qvec[0]) * p[0] * q[0], (1 - qvec[0]) * p[0] * (1 - q[0]), (1 - qvec[0]) * (1 - p[0]) * q[0],
                   (1 - qvec[0]) * (1 - p[0]) * (1 - q[0])],
                  [qvec[1] * p[1] * q[1], qvec[1] * p[1] * (1 - q[1]), qvec[1] * (1 - p[1]) * q[1],
                   qvec[1] * (1 - p[1]) * (1 - q[1]),
                   (1 - qvec[1]) * p[1] * q[1], (1 - qvec[1]) * p[1] * (1 - q[1]), (1 - qvec[1]) * (1 - p[1]) * q[1],
                   (1 - qvec[1]) * (1 - p[1]) * (1 - q[1])],
                  [qvec[2] * p[2] * q[2], qvec[2] * p[2] * (1 - q[2]), qvec[2] * (1 - p[2]) * q[2],
                   qvec[2] * (1 - p[2]) * (1 - q[2]),
                   (1 - qvec[2]) * p[2] * q[2], (1 - qvec[2]) * p[2] * (1 - q[2]), (1 - qvec[2]) * (1 - p[2]) * q[2],
                   (1 - qvec[2]) * (1 - p[2]) * (1 - q[2])],
                  [qvec[3] * p[3] * q[3], qvec[3] * p[3] * (1 - q[3]), qvec[3] * (1 - p[3]) * q[3],
                   qvec[3] * (1 - p[3]) * (1 - q[3]),
                   (1 - qvec[3]) * p[3] * q[3], (1 - qvec[3]) * p[3] * (1 - q[3]), (1 - qvec[3]) * (1 - p[3]) * q[3],
                   (1 - qvec[3]) * (1 - p[3]) * (1 - q[3])],
                  [qvec[4] * p[4] * q[4], qvec[4] * p[4] * (1 - q[4]), qvec[4] * (1 - p[4]) * q[4],
                   qvec[4] * (1 - p[4]) * (1 - q[4]),
                   (1 - qvec[4]) * p[4] * q[4], (1 - qvec[4]) * p[4] * (1 - q[4]), (1 - qvec[4]) * (1 - p[4]) * q[4],
                   (1 - qvec[4]) * (1 - p[4]) * (1 - q[4])],
                  [qvec[5] * p[5] * q[5], qvec[5] * p[5] * (1 - q[5]), qvec[5] * (1 - p[5]) * q[5],
                   qvec[5] * (1 - p[5]) * (1 - q[5]),
                   (1 - qvec[5]) * p[5] * q[5], (1 - qvec[5]) * p[5] * (1 - q[5]), (1 - qvec[5]) * (1 - p[5]) * q[5],
                   (1 - qvec[5]) * (1 - p[5]) * (1 - q[5])],
                  [qvec[6] * p[6] * q[6], qvec[6] * p[6] * (1 - q[6]), qvec[6] * (1 - p[6]) * q[6],
                   qvec[6] * (1 - p[6]) * (1 - q[6]),
                   (1 - qvec[6]) * p[6] * q[6], (1 - qvec[6]) * p[6] * (1 - q[6]), (1 - qvec[6]) * (1 - p[6]) * q[6],
                   (1 - qvec[6]) * (1 - p[6]) * (1 - q[6])],
                  [qvec[7] * p[7] * q[7], qvec[7] * p[7] * (1 - q[7]), qvec[7] * (1 - p[7]) * q[7],
                   qvec[7] * (1 - p[7]) * (1 - q[7]),
                   (1 - qvec[7]) * p[7] * q[7], (1 - qvec[7]) * p[7] * (1 - q[7]), (1 - qvec[7]) * (1 - p[7]) * q[7],
                   (1 - qvec[7]) * (1 - p[7]) * (1 - q[7])]])

    # m_det = np.array(
    #     [[(-1 + qvec[0] * p[0] * q[0]), (-1 + qvec[0] * p[0]), (-1 + qvec[0] * q[0]), qvec[0] * (1 - p[0]) * (1 - q[0]),
    #       (1 - qvec[0]) * p[0] * q[0], (-1 + p[0]), (-1 + q[0]), f[0]],
    #      [qvec[1] * p[1] * q[1], (-1 + qvec[1] * p[1]), qvec[1] * q[1], qvec[1] * (1 - p[1]) * (1 - q[1]),
    #       (1 - qvec[1]) * p[1] * q[1], (-1 + p[1]), q[1], f[1]],
    #      [qvec[2] * p[2] * q[2], qvec[2] * p[2], (-1 + qvec[2] * q[2]), qvec[2] * (1 - p[2]) * (1 - q[2]),
    #       (1 - qvec[2]) * p[2] * q[2], p[2], (-1 + q[2]), f[2]],
    #      [qvec[3] * p[3] * q[3], qvec[3] * p[3], qvec[3] * q[3], (-1 + qvec[3] * (1 - p[3]) * (1 - q[3])),
    #       (1 - qvec[3]) * p[3] * q[3], p[3], q[3], f[3]],
    #      [qvec[4] * p[4] * q[4], qvec[4] * p[4], qvec[4] * q[4], qvec[4] * (1 - p[4]) * (1 - q[4]),
    #       (-1 + (1 - qvec[4]) * p[4] * q[4]), (-1 + p[4]), (-1 + q[4]), f[4]],
    #      [qvec[5] * p[5] * q[5], qvec[5] * p[5], qvec[5] * q[5], qvec[5] * (1 - p[5]) * (1 - q[5]),
    #       (1 - qvec[5]) * p[5] * q[5], (-1 + p[5]), q[5], f[5]],
    #      [qvec[6] * p[6] * q[6], qvec[6] * p[6], qvec[6] * q[6], qvec[6] * (1 - p[6]) * (1 - q[6]),
    #       (1 - qvec[6]) * p[6] * q[6], p[6], (-1 + q[6]), f[6]],
    #      [qvec[7] * p[7] * q[7], qvec[7] * p[7], qvec[7] * q[7], qvec[7] * (1 - p[7]) * (1 - q[7]),
    #       (1 - qvec[7]) * p[7] * q[7], p[7], q[7], f[7]]])
    # return m, m_det
    return m


def determinant(m_det):
    return np.linalg.det(m_det)


def average_game_keys(s_l, a_l):
    keys_value = []
    for s in s_l:
        for a_i in a_l:
            for a_j in a_l:
                keys_value.append((s, (a_i, a_j)))
    return keys_value


def calc_payoff(qvec, p, q, f_p, f_q):
    m = build_markov_chain(qvec, p, q)
    null_matrix = np.transpose(m) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = f_q.reshape(f_q.size, 1).transpose()
    r_p = np.dot(f_p, v)[0]
    r_q = np.dot(f_q, v)[0]
    v = v.flatten()
    return v, r_p, r_q

def average_payoff_matrix(strategy, qvec, f_p, f_q, keys_value):
    p0, p1, q0, q1 = strategy
    r_dict = {}
    for i in range(len(keys_value)):
        r_dict[keys_value[i]] = [0.0, 0.0]
    for key_item in keys_value:
        s = key_item[0]
        a_p = key_item[1][0]
        a_q = key_item[1][1]
        p = [p0, p0, p0, p0, p1, p1, p1, p1]
        q = [q0, q0, q0, q0, q1, q1, q1, q1]
        for _ in range(4):
            p[s * 4 + _] = a_p
            q[s * 4 + _] = a_q
        v, r_p, r_q = calc_payoff(qvec, p, q, f_p, f_q)
        r_dict[key_item][0] = r_p[0]
        r_dict[key_item][1] = r_q[0]
    print(r_dict)
    return r_dict

def evolve(strategy, step_size, qvec, f_p, f_q, s_l, a_l):
    p0, p1, q0, q1 = strategy
    p = [p0, p0, p0, p0, p1, p1, p1, p1]
    q = [q0, q0, q0, q0, q1, q1, q1, q1]
    v, r_p, r_q = calc_payoff(qvec, p, q, f_p, f_q)
    v0 = np.sum(v[0:4])
    v1 = np.sum(v[4:])

    dp0 = (calc_payoff(qvec, s00, q, f_p, f_q)[1][0] - r_p[0]) * p0 * v0
    dp1 = (calc_payoff(qvec, s01, q, f_p, f_q)[1][0] - r_p[0]) * p1 * v1
    dq0 = (calc_payoff(qvec, p, s10, f_p, f_q)[2][0] - r_q[0]) * q0 * v0
    dq1 = (calc_payoff(qvec, p, s11, f_p, f_q)[2][0] - r_q[0]) * q1 * v1
    print(dp0, dp1, dq0, dq1)
    p0 = valid_s(p0 + dp0 * step_size)
    p1 = valid_s(p1 + dp1 * step_size)
    q0 = valid_s(q0 + dq0 * step_size)
    q1 = valid_s(q1 + dq1 * step_size)
    return [p0, p1, q0, q1]


if __name__ == '__main__':
    # t = np.arange(0, 150000)
    # step_length = 0.001
    # # qvec = [0.7, 0.9, 0.3, 0.5, 0.5, 0.7, 0.1, 0.3]
    # # qvec = [0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4]
    qvec = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    # qvec = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    p0 = 0.6; p1=0.1; q0 = 0.3; q1 = 0.1
    strategy = [p0, p1, q0, q1]
    b = 2.0; c = 1.0
    f_p = np.array([b-c, -c, b, 0, b-c, -c, b, 0])
    f_q = np.array([b-c, b, -c, 0, b-c, b, -c, 0])
    # f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
    # f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
    # d = []
    # d.append(strategy)
    # for _ in t:
    #     print(_, strategy)
    #     strategy = evolve(strategy, step_length, qvec, f_p, f_q)
    #     d.append(strategy)
    s_l = [0, 1]
    a_l = [1, 0]
    keys_value = average_game_keys(s_l, a_l)
    print(keys_value)
    average_payoff_matrix(strategy, qvec, f_p, f_q, keys_value)





