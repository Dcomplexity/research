from scipy import stats
import numpy as np

n = 210
N = 400
m = 21
k = 10
p = n / N

x_w = np.arange(0, k+1, 1)
plist_w = stats.binom.pmf(x_w, k, p)
p_w = np.sum(plist_w[k // 2 + 1 :])

x = np.arange(0, m + 1, 1)
plist = stats.binom.pmf(x, m, p_w)
decision_w = np.sum(plist[m // 2 + 1 :])

print(decision_w)
