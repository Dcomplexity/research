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

def calc_payoff(m, f_p, f_q):
    null_matrix = np.transpose(m) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = f_q.reshape(f_q.size, 1).transpose()
    r_p = np.dot(f_p, v)[0]
    r_q = np.dot(f_q, v)[0]
    v = v.flatten()
    return v, r_p, r_q

if __name__ == '__main__':
    # qvec = [0.7, 0.9, 0.3, 0.5, 0.5, 0.7, 0.1, 0.3]
    qvec = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    # p = np.random.random(8)
    p = [0, 0, 0, 0, 0, 0, 0, 0]
    q = np.random.random(8)
    b = 2.0; c = 1.0
    f_p = np.array([b-c, -c, b, 0, b-c, -c, b, 0])
    f_q = np.array([b-c, b, -c, 0, b-c, b, -c, 0])
    m = build_markov_chain(qvec, p, q)
    v, r_p, r_q = calc_payoff(m, f_p, f_q)
    print(np.sum(v[0:4]))
    print(r_p, r_q)






