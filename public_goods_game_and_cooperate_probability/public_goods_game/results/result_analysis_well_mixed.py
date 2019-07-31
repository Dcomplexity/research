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
filename_imitation = dirname + "results_pgg_well_mixed_imitation.csv"
filename_rl = dirname + "results_pgg_well_mixed_rl.csv"
data_imitation = pd.read_csv(filename_imitation, index_col=0)
data_rl = pd.read_csv(filename_rl, index_col=0)
data = pd.concat([data_imitation['co_frac'], data_rl], axis=1)
data.columns = {'imitation', 'rl'}
plt.figure()
data.plot()
plt.xlabel(r'$\gamma$')
plt.ylabel(r'Fraction of cooperators')
plt.show()
