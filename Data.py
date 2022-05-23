# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:52:53 2021

@author: axmao2-c
"""

import pickle
import numpy as np
import pandas as pd
from math import ceil
# from sklearn.preprocessing import StandardScaler


# #Obtain the whole dataset with normalization
# path = 'C:\\Workplace\\Data\\Acc_Gyr_Data.csv'
# df_raw = pd.read_csv(path)
# df_raw = df_raw.drop(['sample_index'], axis=1)
# feature_columns = df_raw.columns[0:6]
# #data standardization
# for i in feature_columns:
#     s_raw = StandardScaler().fit_transform(df_raw[i].values.reshape(-1,1))
#     df_raw[i]  = s_raw.reshape(-1)
# df_raw.to_csv("C:\\Users\\axmao2-c\\OneDrive - City University of Hong Kong\\Desktop\\Fed-SL\\Codes\\Acc_Gyr_Norm_Data.csv", index_label="sample_index")

def Data_Segm(df_data, single=True, tri=False):
  segments,counts = np.unique(df_data["segment"], return_counts = True)
  samples = []
  labels = []
  for s in segments:
    data_segment = df_data[df_data['segment'] == s]
    sample_persegm = []
    for j in range(0,len(data_segment),100):
      temp_sample = data_segment[['Ax','Ay','Az','Gx','Gy','Gz']].iloc[j:j+200,:].values
      #temp_sample = data_segment[['Gx','Gy','Gz']].iloc[j:j+200,:].values
      if len(temp_sample) == 200:
        sample_persegm.append(temp_sample)
    samples.append(sample_persegm)
    labels.append(list(set(data_segment['label']))[0])

  samples_all = []
  labels_all = []
  for i in range(len(labels)):
    if single:
      for s in samples[i]:
        samples_all.append([s])
        labels_all.append(labels[i])
    if tri:
      for j in range(len(samples[i])):
        if (j+2) < len(samples[i]):
          samples_all.append([samples[i][j], samples[i][j+1], samples[i][j+2]])
          labels_all.append(labels[i])
  
  return samples_all, labels_all


#Get training data, validation data, and test data
def get_data(subject = 2):
    
    path = 'C:\\Users\\axmao2-c\\OneDrive - City University of Hong Kong\\Desktop\\Fed-SL\\Codes\\Acc_Gyr_Norm_Data.csv'
    df_train_raw = pd.read_csv(path)
    df_train_raw = df_train_raw.drop(['sample_index'], axis=1)

    #数值对应6中行为['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
    df_train_raw.replace({'grazing':0,'eating':0,'galloping-natural':1,'galloping-rider':1,'standing':2,'trotting-rider':3,'trotting-natural':3,'walking-natural':4,'walking-rider':5},inplace = True)
    df_data = df_train_raw[df_train_raw['subject']==subject]
    
    #Segmentation
    samples, labels = Data_Segm(df_data, single=True, tri=False)
    
    samples_arr = np.array(samples)
    labels_arr = np.array(labels)
    
    return samples_arr, labels_arr

def Get_pkl_file(client_n, data_x, data_y, train = True):
    
    if train == True:
        ##shuffle the training set
        indices = np.arange(data_y.shape[0])
        np.random.shuffle(indices)
        shf_x = data_x[indices]
        shf_y = data_y[indices]
        
        ###get train.part.pkl
        size = ceil((shf_x.shape[0])/10)
        name_n = 0
        for i in range(0, shf_x.shape[0], size):
            save_path = 'C:\\Users\\axmao2-c\\OneDrive - City University of Hong Kong\\Desktop\\Fed-SL\\Codes\\Datasets_Fed_SL\\Client_' + str(client_n) +'\\train_part' + str(name_n) + '.pkl'
            temp = (shf_x[i:i+size], shf_y[i:i+size])
            # print(i)
            pickle.dump(temp, open(save_path, "wb" ))
            name_n += 1
    else:
        ###get test.part.pkl
        pickle.dump((data_x, data_y), open('C:\\Users\\axmao2-c\\OneDrive - City University of Hong Kong\\Desktop\\Fed-SL\\Codes\\Datasets_Fed_SL\\Test' +'\\test.pkl', "wb" ))
        
    return

###Load Dataset///Each equine represents a client
###Client_1
X_train_1, Y_train_1 = get_data(subject = 2)
Get_pkl_file(1, X_train_1, Y_train_1, train = True)
###Client_2
X_train_2, Y_train_2 = get_data(subject = 3)
Get_pkl_file(2, X_train_2, Y_train_2, train = True)
###Client_3
X_train_3, Y_train_3 = get_data(subject = 7)
Get_pkl_file(3, X_train_3, Y_train_3, train = True)
###Client_4
X_train_4, Y_train_4 = get_data(subject = 8)
Get_pkl_file(4, X_train_4, Y_train_4, train = True)
###Client_5
X_train_5, Y_train_5 = get_data(subject = 11)
Get_pkl_file(5, X_train_5, Y_train_5, train = True)
###Test set
X_test, Y_test = get_data(subject = 14)
Get_pkl_file(0, X_test, Y_test, train = False)

