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

filename_imitation = dirname + "results_pgg_well_mixed_imitation_punishment.csv"
filename_rl = dirname + "results_pgg_well_mixed_rl_punishment.csv"
data_imitation = pd.read_csv(filename_imitation, index_col=0)
data_rl = pd.read_csv(filename_rl, index_col=0)
fig, axs = plt.subplots(2, 1, figsize=(6, 8))
data_imitation.plot(ax=axs[0], title="imitation")
data_rl.plot(ax=axs[1], title='rl')
plt.show()
print(data_rl)