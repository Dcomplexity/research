import numpy as np
import pandas as pd

def build_markov_chain(p, q, f):
    m = np.array([[(-1 + p[0] * q[0]), (-1 + p[0]), (-1 + q[0]), f[0]],
                  [p[1] * q[2], (-1 + p[1]), q[2], f[1]],
                  [p[2] * q[1], p[2], (-1 + q[1]), f[2]],
                  [p[3] * q[3], p[3], q[3], f[3]]])
    # m = np.array([[f[0], (-1 + p[0]), (-1 + q[0]), (1-p[0])*(1-q[0])],
    #               [f[1] * q[2], (-1 + p[1]), q[2], (1-p[1])*(1-q[1])],
    #               [f[2] * q[1], p[2], (-1 + q[1]), (1-p[2])*(1-q[2])],
    #               [f[3] * q[3], p[3], q[3], (1-p[3])*(1-q[3])-1]])
    return m


def determinant(m):
    return np.linalg.det(m)


if __name__ == '__main__':
    # p = [0.5, 0.5, 0.5, 0.5]
    # q = [0.5, 0.5, 0.5, 0.5]
    p = [11/13, 1/2, 7/26, 0]
    q = [1, 1, 1, 1]
    # p = [1, 1, 0, 0.7]
    # q = [1, 1, 0, 0.7]
    f_p = [3, 0, 5, 1]
    f_q = [3, 5, 0, 1]
    f_1 = [1, 1, 1, 1]
    # m = build_markov_chain(p, q, f)
    r_p = determinant(build_markov_chain(p, q, f_p)) / determinant(build_markov_chain(p, q, f_1))
    r_q = determinant(build_markov_chain(p, q, f_q)) / determinant(build_markov_chain(p, q, f_1))
    print(r_p, r_q)

