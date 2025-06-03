import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import scipy.io as scio
import torch as t
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from model.sample import random_missing_data
from model.sample import apply_missing_data



warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
    
    
class Dataset_1(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None,epoch=0,begin_row=[],mask=None,data_2D=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test1', 'neighbor']
        type_map = {'train': 0,  'test1': 1, 'neighbor':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        if mask is not None:
            self.mask = mask.cpu().numpy()
        else:
            self.mask = None
  
        self.epoch=epoch
        self.root_path = root_path
        self.data_path = data_path       
        self.flag=flag
        self.begin_row=begin_row
        self.n=len(begin_row)
        self.data_name = args.data_name
        if self.data_name == 'Satellite_part_3D'or self.data_name == 'sun_3D':
            self.num_days = args.num_day*60
        else:
            self.num_days = args.num_day*24

        
        if data_2D is not None:
            self.data_2D = data_2D
        else:
            self.data_2D = None
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))

       

        
        begin_row1=0
        if self.flag == 'neighbor':
            if self.data_name == 'Satellite_part_3D' or self.data_name == 'sun_3D':
                df_current = pd.concat([df_raw.iloc[int(row):int(row)+60*self.args.num_day] for row in self.begin_row.tolist()])
                df_data = pd.concat([pd.DataFrame(self.data_2D[int(row):int(row)+60*self.args.num_day].cpu().numpy()) for row in self.begin_row.tolist()])
            else:
                df_current = pd.concat([df_raw.iloc[int(row):int(row)+24*self.args.num_day] for row in self.begin_row.tolist()])
                df_data = pd.concat([pd.DataFrame(self.data_2D[int(row):int(row)+24*self.args.num_day].cpu().numpy()) for row in self.begin_row.tolist()])
        
            data = torch.tensor(df_data.values) if isinstance(df_data, pd.DataFrame) else torch.tensor(df_data)
            data2 = data
            # print('neighbor')
        elif self.flag == 'test1':
            begin_row1=5*self.num_days
            df_current = df_raw.iloc[begin_row1:begin_row1+self.num_days*5]
            df_data = self.data_2D[begin_row1:begin_row1+self.num_days*5]
            data=df_data
            data2=data
        
        else:
            df_current = df_raw.iloc[-self.num_days:]
            df_data = self.data_2D[-self.num_days:]
            
            data = df_data  
            data2=data
        
        data = data.to('cpu').numpy()
        data[np.isnan(data)] = float(0.0)
        data[np.isinf(data)] = float(0.0)

        data2 = data2.to('cpu').numpy() if isinstance(data2, torch.Tensor) else np.array(data2)
        data2[np.isnan(data2)] = float(0.0)
        data2[np.isinf(data2)] = float(0.0)
        

        df_stamp = df_current[['date']] 
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data  
        self.data_y = data
        self.data2=data2
        

       
        self.data_stamp = data_stamp    

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        data3=self.data2[s_begin:s_end]
        if self.mask is not None:
            mask1=self.mask[s_begin:s_end]
        else:
            mask1=self.data2[s_begin:s_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark,mask1,data3
    

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
    






