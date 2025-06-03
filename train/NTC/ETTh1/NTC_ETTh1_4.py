import torch as t 
import os
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import sys
 
from model.NTC import NTC
import h5py
import time
from model.utils_for_train import find_nearest_neighbors
from model.utils_for_train import loss_compare
from model.utils_for_train import get_score
from model.utils_for_train import get_dataslice
from model.utils_for_train import get_data_all
from model.utils_for_train import get_sample_whole
from model.utils_for_train import create_3d_tensor_from_train_list
from model.utils_for_train import set_seed
from model.utils_for_train import normalize_sum_to_one
from model.utils_for_train import load_and_fill_data
from model.utils_for_train import window_slide


 
device = torch.device("cuda:0")
print(torch.cuda.is_available())


 
class CustomWeightedLoss(nn.Module):
    def __init__(self):
        super(CustomWeightedLoss, self).__init__()

    def forward(self, y_pred, y_true,W=[],L=[]):
         
        loss = 0

         
        if len(W) ==0:  
             
            loss = abs((y_pred-y_true))
             
        
         
        else:    
            for i in range(len(W)):
                loss += L[i]* W[i]
        
        return loss
    
 
criterion = CustomWeightedLoss()

 
seed_value = 42
set_seed(seed_value)

 
g = torch.Generator()
g.manual_seed(seed_value)

 
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_NTC_with_Predata(lr,WX,batch_size,sample_rate,embedding_dim,epoches,nc,count,num_day,window_size,stride,data_name,data_preprocess_full_name,data_pth,data_preprocess_full_pth,result_pth,save_dir_pth): 
    num_epoches = epoches
    sample_rate=sample_rate
    fz,fm=1,100000
    
     
    m = scio.loadmat(data_pth)
     
     
    data=t.Tensor(m[data_name]).to(device) 
    datal=data
     
    data=data[:,:,-num_day:]
    

     
     
    data_file_path =  data_pth
    data_preprocess_full_pth = data_preprocess_full_pth
     
    data_name = data_name
    data_preprocess_full_name = data_preprocess_full_name
     
    start_row = 0
    end_row = datal.shape[2]-num_day         
     
     
    m2 = scio.loadmat(data_preprocess_full_pth)
    matrix_past_fill=t.Tensor(m2[data_preprocess_full_name]).to(device)
    matrix_past_fill = matrix_past_fill.cpu().numpy()
    

     
    time_sample_matrix_fill =window_slide(matrix_past_fill,datal.shape[2],window_size=window_size,stride=stride)
    time_sample_matrix_fill = time_sample_matrix_fill.astype(np.float32)
     
    matrix_past = m[data_name]
    matrix_past = matrix_past[:, :, start_row:end_row]
    time_sample_matrix=window_slide(matrix_past,datal[2],window_size=window_size,stride=stride)
    time_sample_matrix = time_sample_matrix.astype(np.float32)

    
    mi = data.min()

     
    model=NTC(data.shape[0],data.shape[1],data.shape[2],embedding_dim,channels=100)
    print(model)
    params=list(model.parameters())

     
    model=model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))  

     
     
    
    data_erf=data
    
     
    save_dir = save_dir_pth
    os.makedirs(save_dir, exist_ok=True)   

     
    previous_loss = float('inf')


    for epoch in range(num_epoches):

         
        optimizer.zero_grad()

        train_list,test_list=[],[]   
        model.train()    
        
        sum_loss_X1=0    
        

        W=[]     
        L=[]     

        with open(result_pth,'a+') as f:
            print('epoch = ',epoch,file=f)
        
         
        print("............start training............")
        print("............X1............")
         

        for k in range(data_erf.shape[2]):   
            seed_num = int(epoch+1)
            traink,testk=get_sample_whole(data_erf,k,sample_rate,seed_num)
            test_list=testk
            train_list+=traink
            train_loader = t.utils.data.DataLoader(traink, batch_size=batch_size, num_workers=0, shuffle=True,worker_init_fn=seed_worker,generator=g,drop_last=True)   

             
            if (k+1)%5==0 and (k+1)>=5:
                print('[{}]/[{}]'.format(k+1,data_erf.shape[2]))
                   
            running_loss_X1=0    
                                   
            for i,data_i in enumerate(train_loader,0):    
                o_inputs,d_inputs,t_inputs,scores= data_i
                scores=scores.float()
                t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)
    
                outputs = model(o_inputs, d_inputs, t_inputs,is_history=False).to(device)    
                outputs= torch.squeeze(outputs, 1)   
                
                 
                loss = criterion(outputs,scores,W,L)

                 
                running_loss_X1+=loss.sum()
        
             
            sum_loss_X1+=running_loss_X1/len(train_loader.dataset)
             
            del traink, testk, train_loader,running_loss_X1
        
        sum_loss_X1=sum_loss_X1/data_erf.shape[2]

         
         
         
             
 

        
         
        print("............X2............")

         
         
         
         
         
         
         
         
        
         
        data_neigh = create_3d_tensor_from_train_list(train_list, data.shape)


         
        neighbor_matrix,weight_neighbor,topk_idx = find_nearest_neighbors(data_neigh,time_sample_matrix,time_sample_matrix_fill)

         
       
        sum_loss_X2=[]    
          
         
        for n in range(neighbor_matrix.shape[0]):
            data_neigh_tensor = torch.tensor(neighbor_matrix[n,:,:,:]).to(device) 

            train_neigh_list=[]
            sum_loss_X2_N=0
             
            for k in range(data_neigh_tensor.shape[2]):
                 
                data_neigh_slice = data_neigh_tensor[:,:,k]
                index_data_neigh=(data_neigh_slice!=0)
                nonzero_data_neigh = index_data_neigh.nonzero(as_tuple=True)
                coordinates_data_neigh = torch.stack(nonzero_data_neigh, dim=1)  
                train_neigh=list(map(lambda x:(x[0],x[1],k,data_neigh_tensor[x[0],x[1],k]),coordinates_data_neigh))    
                train_neigh_list+=train_neigh
                train_neigh_loader=t.utils.data.DataLoader(train_neigh,batch_size=batch_size, num_workers=0, shuffle=True,worker_init_fn=seed_worker,generator=g,drop_last=True)   

                 
                if (k+1)%5==0 and (k+1)>=5:
                    print('[{}]/[{}]'.format(k+1,data_neigh_tensor.shape[2]))

                running_loss_X2_N=0    

                for i,data_i in enumerate(train_neigh_loader,0):   
                    o_inputs,d_inputs,t_inputs,scores= data_i
                    scores=scores.float()
                    t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)
 
                    outputs = model(o_inputs, d_inputs, t_inputs,is_history=True)    
                    
                    outputs= torch.squeeze(outputs, 1)   
                     
                    outputs_mask = ~torch.isnan(outputs)
                     
                    scores_mask = ~torch.isnan(scores)
                     
                    valid_mask = outputs_mask & scores_mask

                     
                    valid_outputs = outputs[valid_mask]
                    valid_scores = scores[valid_mask]
                     
                     
                    loss = criterion(valid_outputs, valid_scores, W, L)                    
                                        
                     
                    running_loss_X2_N+=loss.sum()

                    
                 
                sum_loss_X2_N+=running_loss_X2_N/len(train_neigh_loader.dataset)
                 
                del train_neigh, train_neigh_loader,running_loss_X2_N

             
            sum_loss_X2_N = sum_loss_X2_N/data_neigh_tensor.shape[2]
            sum_loss_X2.append(sum_loss_X2_N)
          
         
        W.append(WX)
        L.append(sum_loss_X1)
        weight_neighbor=weight_neighbor.numpy()
        for i in range(neighbor_matrix.shape[0]):
            W.append(weight_neighbor[i])
            L.append(sum_loss_X2[i])

        W = normalize_sum_to_one(W)
        sum_loss = criterion(y_pred=0,y_true=0,W=W,L=L)
        

         
        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        current_loss = sum_loss.item()

        with open(result_pth,'a+') as f:
            print('epoch sum loss = ',current_loss,file=f)
            
 
         
         
        print("............start evaluating single(only X1)............")
        time_evaluate_start = time.time() 
         
        if epoch>=0:
            model.eval()
            print('[{}]/[{}]'.format(epoch+1,num_epoches))
             
            train_eval=t.utils.data.DataLoader(train_list,batch_size=1,shuffle=True)
            test_eval=t.utils.data.DataLoader(test_list,batch_size=1,shuffle=True)

            test_er,test_mae,test_mse,test_rse,test_rmse,test_mape,test_mspe=get_score(dataset=test_eval,model=model,count=3000,sp=data.shape)
             

             
            with open(result_pth,'a+') as f:
                print('sample_rate:',sample_rate,file=f)
                 
                print('nc:',nc,file=f)
                print('embedding_dim(R):',embedding_dim,file=f)
                print('test er:',test_er.item(),file=f)
                print('test_mae:',test_mae.item(),file=f)
                print('test_mse:',test_mse.item(),file=f)
                print('test_rse:',test_rse.item(),file=f)
                print('test_rmse:',test_rmse.item(),file=f)
                print('test_mape:',test_mape.item(),file=f)
                print('test_mspe:',test_mspe.item(),file=f)
                

            time_evaluate_end = time.time()
            time_evaluate = time_evaluate_end - time_evaluate_start
            with open(result_pth,'a+') as f:
                print('evaluate time X1= ',time_evaluate,file=f)
                print('\n',file=f)

         
         
         
         
         
         
         
         
         
         
         
         
         
         

         
         

         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
                
         
         
         
         
         
            
         

         
        previous_loss = current_loss 

        if (epoch + 1) % 100 == 0:
            file_path = os.path.join(save_dir_pth, f'X2_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), file_path)
            print(f"Model X2 saved at {file_path}")

         
        del train_list
        del test_list
        del train_neigh_list
        torch.cuda.empty_cache()   



lr = 0.0001
WX=10    
batch_size=1
sample_rate=1  
embedding_dim=8
epoches=2000
nc=100
count=300
num_day = 1     
window_size = 1
stride = 1
data_name = 'ETTh1_3D'
data_preprocess_full_name = 'data'
data_pth='../matrix_filling_pq/data/ETTh1/ETTh1_3D.mat'
data_preprocess_full_pth = '../matrix_filling_pq/data/data_preprocessing/ETTh1_full.mat'
result_pth='../matrix_filling_pq/result 
save_dir_pth = '../matrix_filling_pq/save_model/NTC/ETTh1/3*/'


train_NTC_with_Predata(lr,WX,batch_size,sample_rate,embedding_dim,epoches,nc,count,num_day,window_size,stride,data_name,data_preprocess_full_name,data_pth,data_preprocess_full_pth,result_pth,save_dir_pth)
        
        
       

    
            
            