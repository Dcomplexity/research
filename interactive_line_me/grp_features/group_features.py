import pandas as pd
import numpy as np
import scipy
from scipy import io
import os


def get_group_data(file_name):
    features_struct = scipy.io.loadmat(file_name)
    raw_ind_features = features_struct['speechStats_forGrp']
    return raw_ind_features


if __name__ == "__main__":
    abs_path = os.path.abspath(os.path.join(os.getcwd(), './'))
    dir_name = abs_path + '/group_data_set/'
    files_list = []
    for root, dirs, files in os.walk(dir_name):
        files_list.append(files)
    grp_data_list = files_list[0][:]
    if ".DS_Store" in grp_data_list:
        grp_data_list.remove('.DS_Store')
    grp_data_list.sort()
    file_name = dir_name + grp_data_list[0]
    result_group_data = get_group_data(file_name)
    print(len(result_group_data[0]))
    print(result_group_data[0][0][1])