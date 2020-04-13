import numpy as np
import pandas as pd
from sympy import Matrix
from scipy.linalg import null_space


def build_markov_chain(qvec, p, q, f):
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

    m_det = np.array(
        [[(-1 + qvec[0] * p[0] * q[0]), (-1 + qvec[0] * p[0]), (-1 + qvec[0] * q[0]), qvec[0] * (1 - p[0]) * (1 - q[0]),
          (1 - qvec[0]) * p[0] * q[0], (-1 + p[0]), (-1 + q[0]), f[0]],
         [qvec[1] * p[1] * q[1], (-1 + qvec[1] * p[1]), qvec[1] * q[1], qvec[1] * (1 - p[1]) * (1 - q[1]),
          (1 - qvec[1]) * p[1] * q[1], (-1 + p[1]), q[1], f[1]],
         [qvec[2] * p[2] * q[2], qvec[2] * p[2], (-1 + qvec[2] * q[2]), qvec[2] * (1 - p[2]) * (1 - q[2]),
          (1 - qvec[2]) * p[2] * q[2], p[2], (-1 + q[2]), f[2]],
         [qvec[3] * p[3] * q[3], qvec[3] * p[3], qvec[3] * q[3], (-1 + qvec[3] * (1 - p[3]) * (1 - q[3])),
          (1 - qvec[3]) * p[3] * q[3], p[3], q[3], f[3]],
         [qvec[4] * p[4] * q[4], qvec[4] * p[4], qvec[4] * q[4], qvec[4] * (1 - p[4]) * (1 - q[4]),
          (-1 + (1 - qvec[4]) * p[4] * q[4]), (-1 + p[4]), (-1 + q[4]), f[4]],
         [qvec[5] * p[5] * q[5], qvec[5] * p[5], qvec[5] * q[5], qvec[5] * (1 - p[5]) * (1 - q[5]),
          (1 - qvec[5]) * p[5] * q[5], (-1 + p[5]), q[5], f[5]],
         [qvec[6] * p[6] * q[6], qvec[6] * p[6], qvec[6] * q[6], qvec[6] * (1 - p[6]) * (1 - q[6]),
          (1 - qvec[6]) * p[6] * q[6], p[6], (-1 + q[6]), f[6]],
         [qvec[7] * p[7] * q[7], qvec[7] * p[7], qvec[7] * q[7], qvec[7] * (1 - p[7]) * (1 - q[7]),
          (1 - qvec[7]) * p[7] * q[7], p[7], q[7], f[7]]])
    return m, m_det

def determinant(m_det):
    return np.linalg.det(m_det)


if __name__ == '__main__':

    # qvec = np.random.random(8)
    # qvec = [0.7, 0.3, 0.9, 0.5, 0.5, 0.1, 0.7, 0.3]
    qvec = [0.7, 0.9, 0.3, 0.5, 0.5, 0.7, 0.1, 0.3]

    # p = np.random.random(8)
    p = np.zeros(8)
    q = np.random.random(8)
    # p = [11/13, 1/2, 7/26, 0, 11/13, 1/2, 7/26, 0]
    # q = [1, 1, 1, 1, 1, 1, 1, 1]

    # f_p = np.array([3, 0, 5, 1, 3, 0, 5, 1])
    f_p = np.array([3, 1, 4, 2, 7, 5, 8, 6])
    f_q = np.array([3, 4, 1, 2, 7, 8, 5, 6])
    # f_q = np.array([3, 5, 0, 1, 3, 5, 0, 1])
    f_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    m_p, m_p_det = build_markov_chain(qvec, p, q, f_p)
    m_q, m_q_det = build_markov_chain(qvec, p, q, f_q)
    r_p = determinant(m_p_det) / determinant(build_markov_chain(qvec, p, q, f_1)[1])
    r_q = determinant(m_q_det) / determinant(build_markov_chain(qvec, p, q, f_1)[1])
    null_matrix = np.transpose(m_p) - np.eye(8)
    v = null_space(null_matrix)
    v = v / np.sum(v)
    f_p = f_p.reshape(f_p.size, 1).transpose()
    f_q = f_q.reshape(f_q.size, 1).transpose()
    f1 = np.dot(f_p, v)[0]
    f2 = np.dot(f_q, v)[0]
    v = v.flatten()
    print(v)
    print(f1, f2)
    print(r_p, r_q)