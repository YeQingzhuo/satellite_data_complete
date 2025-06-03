from data_provider.data_loader import Dataset_1
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import os
import scipy.io as scio
import torch as t

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dict = {
    # 'data': Dataset,
    'data': Dataset_1
}



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def data_provider(args,flag,epoch=0,begin_row=[],mask=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    
    data_file_path =   os.path.join(args.root_path, args.data_mat_path)
    data_name = args.data_name
    m = scio.loadmat(data_file_path)
     
    data=t.Tensor(m[data_name]).to(device) 
    
    data_permuted = data.permute(0, 2, 1)
    data_2D = data_permuted.reshape(-1, data.size(1))
    

    
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        epoch=epoch,
        begin_row=begin_row,
        mask = mask,
        data_2D = data_2D
    )
    # print(flag, len(data_set))
    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader

