import pandas as pd
import numpy as np
import scipy
from scipy import io
import os


def get_ind_data(file_name):
    features_struct = scipy.io.loadmat(file_name)
    raw_ind_features = features_struct['cln_speechMsk_forInd']
    raw_ind_features = np.array(raw_ind_features)
    ind_features = []
    for i in raw_ind_features:
        ind_features.append(i[0].reshape(-1))
    ind_features = np.transpose(ind_features)
    ind_features_df = pd.DataFrame(ind_features)
    return ind_features_df
    
if __name__ == "__main__":
    dir_name = './ind_data_set/'
    files_list = []
    for root, dirs, files in os.walk(dir_name):
        files_list.append(files)
    files_list[0].sort()
    files_list[0].remove('.DS_Store')

    average_speak_time = {}
    for ind_files in files_list[0]:
        file_name = dir_name + ind_files
        print(file_name[-7:-4])
        ind_data = get_ind_data(file_name)
        average_speak_time[file_name[-7:-4]] = ind_data.mean()
    average_speak_time = pd.DataFrame(average_speak_time).T
    average_speak_time.to_csv('./data_process_result/average_speak_time.csv')