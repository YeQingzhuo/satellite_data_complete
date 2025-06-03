import torch as t 
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import sys
  
from model.Costco import CostCo
import h5py
import nanopq
import time
import random
from model.sample import random_missing_data
from model.sample import whole_missing_data
from model.sample import apply_missing_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


  
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



  
  
  
  

  
  
  
  
  
  
  

  
  
  
  
  
  
  

  
  
  
  
  
  
  

  


  
def filling_average(matrix_past):
    T = matrix_past
    I, J, K = T.shape

      
    T[np.isnan(T)] = 0
    T[np.isinf(T)] = 0

      
    for i in range(I):
        for j in range(K):
            non_zero_mask = (T[i, :, j] != 0)
            if non_zero_mask.any():
                avg = np.mean(T[i, :, j][non_zero_mask])
                fill_mask = (T[i, :, j] == 0)
                T[i, :, j][fill_mask] = avg

    return T




  
  
  
  
  
  
def window_slide(matrix_past, total_size, window_size, stride):
    num_windows = (total_size - window_size) // stride + 1    
    dim1, dim2, dim3 = matrix_past.shape    
    time_sample_matrix = np.array([matrix_past[:, :, i:i + window_size] for i in range(0, dim3 - window_size + 1, stride)])    
    return time_sample_matrix



def window_slide_1(matrix_past, total_size, window_size, stride):
    num_windows = (total_size - window_size) // stride + 1    
    dim1, dim2, dim3 = matrix_past.shape    
    time_sample_matrix = np.array([matrix_past[ i:i + window_size,:, :] for i in range(0, dim1 - window_size + 1, stride)])    
    return time_sample_matrix





  
def get_sample_matrix(matrix_new):
    sample_matrix = np.zeros(matrix_new.shape)
    for a in range(matrix_new.shape[0]):
        for b in  range(matrix_new.shape[1]):
            for c in range(matrix_new.shape[2]):
                if matrix_new[a][b][c] != 0 :
                    sample_matrix[a][b][c] = 1
    return sample_matrix




  
def multiply_matrix(time_sample_matrix,sample_matrix):
    matrix_new = time_sample_matrix
    for i in range(time_sample_matrix.shape[0]):
        matrix_new[i,:,:,:] = np.multiply(sample_matrix,time_sample_matrix[i,:,:,:])
    return matrix_new



  
def create_3d_tensor_from_train_list(train_list, data_shape):
    """
    从 train_list 创建三维张量
    参数:
    train_list (list): 包含 (tensor1, tensor2, scalar, tensor3) 的列表
    data_shape (tuple): 目标三维张量的形状
    返回:
    data_neigh (torch.Tensor): 填充后的三维张量
    """
    data_neigh = torch.zeros(data_shape, device='cuda:0')
    
    for item in train_list:
        tensor1, tensor2, scalar, tensor3 = item
        a, b, c = tensor1.item(), tensor2.item(), scalar
        data_neigh[a, b, c] = tensor3

      
    data_neigh_np = data_neigh.cpu().numpy()
    
    return data_neigh_np





  
def load_and_fill_data(file_path, data_name, start_row, end_row):
      
    m = scio.loadmat(file_path)
      
    matrix = m[data_name]
      
    matrix_past = matrix[:, :, start_row:end_row]
      
    matrix_past = filling_average(matrix_past)

    return matrix_past

  
def load_and_fill_data_1(file_path, data_name, start_row, end_row):
      
    m = scio.loadmat(file_path)
      
    matrix = m[data_name]
      
    matrix_past = matrix[start_row:end_row:, : ]
      
    matrix_past = filling_average(matrix_past)

    return matrix_past






  
def find_nearest_neighbors(matrix_new,time_sample_matrix,time_sample_matrix_fill):
      
    matrix_new = matrix_new.astype(np.float32)
      
      
      
      
      
      
      
      
      
      
    M = time_sample_matrix_fill.shape[2]
      
    train_input = time_sample_matrix_fill.transpose((0,2,1,3)).reshape((int(time_sample_matrix_fill.shape[0]),-1))
      
      
    pq = nanopq.PQ(M,Ks=8)    
      
    print('...Start training codewords...')
    pq.fit(train_input)
      
    print('...Start Quantization...')
    matrix_past_train_code = pq.encode(train_input)
      
      
    query = matrix_new.transpose((1,0,2)).reshape((-1))
    DistanceTable = pq.dtable(query)
    dists = DistanceTable.adist(matrix_past_train_code)
    dists = torch.Tensor(dists)
      
    num_neighbor = 2
      
    min_values, topk_idxs = torch.topk(dists, num_neighbor, dim=-1, largest=False, sorted=True)
    print("min_values shape= ",min_values.shape)
    print("topk_idxs shape= ",topk_idxs.shape)
      
      
    distance_sum = torch.sum(min_values)
      
    weight_neighbor = min_values / distance_sum
      
    neighbor_matrix = time_sample_matrix[topk_idxs,:,:,:]
    if topk_idxs.numel() == 1:
        neighbor_matrix = np.expand_dims(neighbor_matrix, axis=0)
    return neighbor_matrix,weight_neighbor,topk_idxs




  
def loss_compare(loss1,loss2):
    if (abs(loss1-loss2)<pow(10,-5)):
        return False
    else:
        return True
    




  
def get_score(dataset,model,count,sp):
    out=t.zeros(sp)
    score=t.zeros(sp)  
    count=len(dataset)
    
    with t.no_grad():
        for i,data in enumerate(dataset,0):
            if i==count:
                break
            o_inputs, d_inputs, t_inputs, scores = data
            scores = scores.float()
            t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)
            model=model.to(device)
            outputs = model(o_inputs, d_inputs, t_inputs,is_history=False).squeeze()

            if outputs.item()==0:
                print('o_inputs=,d_inputs=,t_inputs=,scores=',o_inputs,d_inputs,t_inputs,scores)
            
              
            out[o_inputs,d_inputs,t_inputs]=outputs.item()
            score[o_inputs,d_inputs,t_inputs]=scores.item()

          
        er=t.norm(out-score)/t.norm(score)
        mae=t.sum(t.abs(out-score))/count
        mse= t.sum((out - score).pow(2)) / count
        rse=t.sqrt(t.sum((out-score).mul(out-score))/t.sum((score-score.mean()).mul(score-score.mean())))
        rmse=t.sqrt(t.sum((out-score).mul(out-score))/count)       
        mape=(t.sum(t.abs((out-score)/(score+0.000001)))/count)
        mspe = (t.sum(((out - score) / (score + 0.000001)).pow(2)) / count) 
        
    return er,mae,mse,rse,rmse,mape,mspe



  
def get_score_NTF(dataset,model,count,sp):
    out=t.zeros(sp)
    score=t.zeros(sp)

    if count<=len(dataset):
        pass
    else:
        count=len(dataset)
    with t.no_grad():
        for i,data in enumerate(dataset,0):
            if i==count:
                break
            o_inputs, d_inputs, t_inputs, scores = data
            scores = scores.float()
            t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)
            t_inputs_period = idx2seq(d_inputs.cpu().numpy(), period=48)
            model=model.to(device)
            outputs = model(o_inputs, d_inputs, t_inputs,t_inputs_period,is_history=False).squeeze()

            if outputs.item()==0:
                print('o_inputs=,d_inputs=,t_inputs=,scores=',o_inputs,d_inputs,t_inputs,scores)
            
              
            out[o_inputs,d_inputs,t_inputs]=outputs.item()
            score[o_inputs,d_inputs,t_inputs]=scores.item()

          
          
          
          
          
        er=t.norm(out-score)/t.norm(score)
        mae=t.sum(t.abs(out-score))/count
        mse= t.sum((out - score).pow(2)) / count
        rse=t.sqrt(t.sum((out-score).mul(out-score))/t.sum((score-score.mean()).mul(score-score.mean())))
        rmse=t.sqrt(t.sum((out-score).mul(out-score))/count)       
        mape=(t.sum(t.abs((out-score)/(score+0.000001)))/count)
        mspe = (t.sum(((out - score) / (score + 0.000001)).pow(2)) / count) 
        
      
    return er,mae,mse,rse,rmse,mape,mspe




def get_dataslice(data,p,k,sample_rate,data_raw):
      
    p=p[:,:,k]
    index_train=((p<=sample_rate) & (data_raw[:,:,k]!=0))
    index_train=index_train.nonzero()
    index_test=((p>sample_rate) & (data_raw[:,:,k]!=0)).nonzero()

    while len(index_train)==0 or len(index_test)==0:
            p=t.rand_like(data[:,:,k])
            index_train=((p<=sample_rate) & (data_raw!=0)).nonzero()
            index_test=((p>sample_rate) & (data_raw!=0)).nonzero()

    train=list(map(lambda x:(x[0],x[1],k,data[x[0],x[1],k]),index_train))
    test=list(map(lambda x:(x[0],x[1],k,data[x[0],x[1],k]),index_test))

    return train,test



  
def get_sample(data,k,sample_rate,seed_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slice = data[:, :, k].to('cpu').numpy()    

      
    slice[slice == 0.0] = float('nan')
    slice[np.isinf(slice)] = float('nan')
    slice = torch.from_numpy(slice).to(device)

      
    i=int(sample_rate*10)
    index_train,index_test= random_missing_data(slice,seed_num,i)

      
    train_list = apply_missing_data(slice, index_test).to(device)
    test_list = apply_missing_data(slice, index_train).to(device)

      
    index_train = (train_list != 0) & (~torch.isnan(train_list))
    nonzero_train = index_train.nonzero(as_tuple=True)
    coordinates_train = torch.stack(nonzero_train, dim=1)
   
    index_test = (test_list != 0) & (~torch.isnan(test_list))
    nonzero_test = index_test.nonzero(as_tuple=True)
    coordinates_test = torch.stack(nonzero_test, dim=1)

    if nonzero_train[0].numel() > 0:
        train = list(map(lambda x: (x[0], x[1], k, data[x[0], x[1], k]), coordinates_train))
    else:
        train = []

    if nonzero_test[0].numel() > 0:
        test = list(map(lambda x: (x[0], x[1], k, data[x[0], x[1], k]), coordinates_test))
    else:
        test = []

    return train,test



  
def get_sample_whole(data,k,sample_rate,seed_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slice = data[:, :, k].to('cpu').numpy()    

      
    slice[slice == 0.0] = float('nan')
    slice[np.isinf(slice)] = float('nan')
    slice = torch.from_numpy(slice).to(device)

      
    i=int(sample_rate*10)
    index_train,index_test= whole_missing_data(slice,seed_num,i)

      
    train_list = apply_missing_data(slice, index_test).to(device)
    test_list = apply_missing_data(slice, index_train).to(device)

      
    index_train = (train_list != 0) & (~torch.isnan(train_list))
    nonzero_train = index_train.nonzero(as_tuple=True)
    coordinates_train = torch.stack(nonzero_train, dim=1)
   
    index_test = (test_list != 0) & (~torch.isnan(test_list))
    nonzero_test = index_test.nonzero(as_tuple=True)
    coordinates_test = torch.stack(nonzero_test, dim=1)

    if nonzero_train[0].numel() > 0:
        train = list(map(lambda x: (x[0], x[1], k, data[x[0], x[1], k]), coordinates_train))
    else:
        train = []

    if nonzero_test[0].numel() > 0:
        test = list(map(lambda x: (x[0], x[1], k, data[x[0], x[1], k]), coordinates_test))
    else:
        test = []

    return train,test


def get_data_all(data,sample_rate,data_raw):
      
    p=t.rand_like(data)
    index_train=((p<=sample_rate) & (data_raw!=0)).nonzero()
    index_test=((p>sample_rate) & (data_raw!=0)).nonzero()


    while len(index_test)==0 or len(index_train)==0:
        p=t.rand_like(data)
        index_train=((p<=sample_rate) & (data_raw!=0)).nonzero()
        index_test=((p>sample_rate) & (data_raw!=0)).nonzero()

    train=list(map(lambda x:(x[0],x[1],x[2],data[x[0],x[1],x[2]]),index_train))
    test=list(map(lambda x:(x[0],x[1],x[2],data[x[0],x[1],x[2]]),index_test))
    return train,test





  
def normalize_sum_to_one(arr):
    """
    将数组中的数值归一化，使其总和为1。
    
    参数:
    arr (numpy.ndarray): 输入数组
    
    返回:
    numpy.ndarray: 归一化后的数组
    """
    total_sum = np.sum(arr)
    
      
    if total_sum == 0:
        return np.zeros_like(arr)
    
    normalized_arr = arr / total_sum
    return normalized_arr



  
def idx2seq(index, period=5):
    seq = []
    for k in index:
        tmp_seq = torch.arange(k - period, k)
        seq.append(tmp_seq)
    seq = torch.stack(seq)
    seq = torch.clamp(seq, min=0)
    return seq