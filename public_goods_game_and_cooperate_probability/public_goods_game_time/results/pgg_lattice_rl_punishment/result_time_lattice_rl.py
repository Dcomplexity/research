import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import seaborn as sns
import os


abspath = os.path.abspath(os.path.join(os.getcwd(), './'))
dirname = abspath + '/'
for root, dirs, files in os.walk(dirname):
    for filename in files:
        print(filename)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    files.sort()
filename = dirname + "strategy_history_0.4.csv"
data = pd.read_csv(filename, index_col=0)
plt.figure()
data[0:2000].plot()
plt.xlabel(r'time')
plt.show()
# plt.savefig("result_lattice.png")

