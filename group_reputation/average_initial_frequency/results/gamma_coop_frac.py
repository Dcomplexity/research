import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import os
import random
import datetime
import math

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

project_root_dir = '.'

def save_png(fig_id, tight_layout=True):
    path_png = os.path.join(project_root_dir, 'images', fig_id + '.png')
    print('saving png ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path_png, format='png', dpi=300)

abs_path = os.getcwd()
dir_name = os.path.join(abs_path)
file_list = []
for root, dirs, files in os.walk(dir_name):
    file_list.append(files)
file_list[0].sort()
print(file_list[0])

original_file = os.path.join(dir_name, 's_d_pgg_original.csv')
original = pd.read_csv(original_file, index_col=[0])
print(original)
plt.xlabel(r'$\gamma$')
plt.ylabel('fraction of cooperators')
plt.plot(original, label='original')
plt.legend()
save_png("original")
original_punishment_file = os.path.join(dir_name, 's_d_pgg_original_punishment.csv')
original_punishment = pd.read_csv(original_punishment_file, index_col=[0])
original_punishment['c'] = original_punishment['1'] + original_punishment['2']
plt.plot(original_punishment['c'], label='punishment')
plt.legend()
save_png('original_punishment')
competitive_gamma_file = os.path.join(dir_name, 's_d_pgg_competitive_gamma.csv')
competitive_gamma = pd.read_csv(competitive_gamma_file, index_col=[0])
plt.plot(competitive_gamma, label='gamma')
plt.legend()
save_png('competitive_gamma')
competitive_gamma_punishment_file = os.path.join(dir_name, 's_d_pgg_competitive_gamma_punishment.csv')
competitive_gamma_punishment = pd.read_csv(competitive_gamma_punishment_file, index_col=[0])
competitive_gamma_punishment['c'] = competitive_gamma_punishment['1'] + competitive_gamma_punishment['2']
plt.plot(competitive_gamma_punishment['c'], label='gamma_punishment')
plt.legend()
save_png('competitive_gamma_punishment')
