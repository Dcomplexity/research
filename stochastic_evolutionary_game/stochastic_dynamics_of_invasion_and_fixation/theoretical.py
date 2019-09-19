import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd


def calc_phi_k(popu_s, k, beta, r, s, t, p):
    u_value = (r - s -t + p) / 2
    v_value = (-1 * r + s * popu_s - p * popu_s + p) / 2
    numerator = 0
    for i in range(k):
        numerator += math.e ** (-1 * beta * i * (i + 1) * u_value - 2 * beta * i * v_value)
    denominator = 0
    for i in range(popu_s):
        denominator += math.e ** (-1 * beta * i * (i + 1) * u_value - 2 * beta * i * v_value)
    phi_k = numerator / denominator
    return phi_k

if __name__ == '__main__':
    popu_s = 20
    beta = 0.1
    r = 3; s = 1; t = 4; p = 2
    result = []
    for k in range(21):
        print(k)
        phi_k = calc_phi_k(popu_s, k, beta, r, s, t, p)
        result.append(phi_k)
    result = pd.DataFrame({'phi_k': result})
    file_name = './results/%.3f_theoretical_result.csv' % beta
    result.to_csv(file_name)
    print(result)
