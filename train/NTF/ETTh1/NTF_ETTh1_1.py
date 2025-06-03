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
 
from model.NTF import NTF
import h5py
import time
from model.utils_for_train import find_nearest_neighbors
from model.utils_for_train import loss_compare
from model.utils_for_train import get_score_NTF
from model.utils_for_train import get_dataslice
from model.utils_for_train import get_data_all
from model.utils_for_train import get_sample
from model.utils_for_train import set_seed
from model.utils_for_train import idx2seq


 
device = torch.device("cuda:0")
print(torch.cuda.is_available())



 
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
         
        loss = abs((y_pred-y_true))
         
        return loss
    
 
custom_loss = CustomLoss()

 
seed_value = 42
set_seed(seed_value)


 
g = torch.Generator()
g.manual_seed(seed_value)

 
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_NTF(lr,batch_size,sample_rate,embedding_dim,epoches,nc,num_day,data_name,data_pth,result_pth,save_dir_pth): 
    print("1:{}".format(torch.cuda.memory_allocated(0)))
    num_epoches = epoches
    sample_rate=sample_rate
    fz,fm=1,100000
    
    
    m = scio.loadmat(data_pth)
     
     
    data=t.Tensor(m[data_name]).to(device) 
     
    data=data[:,:,-num_day:]
           
    mi = data.min()

     
    model=NTF(data.shape[0],data.shape[1],data.shape[2],embedding_dim,period=48)
    print(model)
    params=list(model.parameters())

     
    model=model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))  

     
     
    
    data_erf=data


     
    os.makedirs(save_dir_pth, exist_ok=True)   

     
    previous_loss = float('inf')

    for epoch in range(num_epoches):
        
         
        optimizer.zero_grad()

        train_list,test_list=[],[]   
        model.train()    
        
        sum_loss_X1=0    

        with open(result_pth,'a+') as f:
            print('epoch = ',epoch,file=f)
        
         
        
        print("............start training............")
        print("............X1............")
         
      
        for k in range(data_erf.shape[2]):   
            seed_num = int(epoch + 1)
            traink, testk = get_sample(data_erf, k, sample_rate, seed_num)
            test_list += testk
            train_list += traink
            train_loader = t.utils.data.DataLoader(traink, batch_size=batch_size, num_workers=0, shuffle=True,worker_init_fn=seed_worker,generator=g,drop_last=True)   

             
            if (k + 1) % 5 == 0 and (k + 1) >= 5:
                print('[{}]/[{}]'.format(k + 1, data_erf.shape[2]))

            running_loss_X1 = 0   

            for i, data_i in enumerate(train_loader, 0):
                o_inputs, d_inputs, t_inputs, scores = data_i
                scores = scores.float()
                t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)
                 
                t_inputs_period = idx2seq(d_inputs.cpu().numpy(), period=48)
                outputs = model(o_inputs, d_inputs, t_inputs,t_inputs_period,is_history=False).to(device)   
                outputs = torch.squeeze(outputs, 1)   

                 
                loss = custom_loss(outputs, scores)

                 
                running_loss_X1 += loss.sum()
                
             
            sum_loss_X1 += running_loss_X1/len(train_loader.dataset)
             
            del traink, testk, train_loader,running_loss_X1
        
         
        sum_loss_X1=sum_loss_X1/data_erf.shape[2]
        sum_loss_X1.backward()
         
        optimizer.step()

        current_loss = sum_loss_X1.item()

        with open(result_pth,'a+') as f:
            print('epoch sum loss = ',current_loss,file=f)

        
        if (epoch + 1) % 100 == 0:
            file_path = os.path.join(save_dir_pth, f'X1_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), file_path)
            print(f"Model X1 saved at {file_path}")

         
        del sum_loss_X1 , loss
        torch.cuda.empty_cache()

        print("epoch=",epoch)
        print("5:{}".format(torch.cuda.memory_allocated(0)))

         
         
        print("............start evaluating single(only X1)............")
        time_evaluate_start = time.time() 
         
        if epoch>=0:
            model.eval()
            with torch.no_grad():
                print('[{}]/[{}]'.format(epoch+1,num_epoches))
                 
                train_eval=t.utils.data.DataLoader(train_list,batch_size=1,num_workers=0,shuffle=True)
                test_eval=t.utils.data.DataLoader(test_list,batch_size=1,num_workers=0,shuffle=True)

                test_er,test_mae,test_mse,test_rse,test_rmse,test_mape,test_mspe= get_score_NTF(dataset=test_eval,model=model,count=3000,sp=data.shape)
                 

                 
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
                print('evaluate time X1= ',time_evaluate)
                with open(result_pth,'a+') as f:
                    print('evaluate time X1= ',time_evaluate,file=f)
                    print('\n',file=f)
            
        print("epoch=",epoch)
        print("eval_used_gpu:{}".format(torch.cuda.memory_allocated(0)))


         
         
         
         
         
         
         
         
         
         
         
         
         
         
         

         
         

         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
                
         
         
         
         
         
            
         

         
        previous_loss = current_loss

         
        del train_list
        del test_list
        torch.cuda.empty_cache()   

        print("epoch=",epoch)
        print("6:{}".format(torch.cuda.memory_allocated(0)))



lr = 0.001
batch_size=2
sample_rate=0.8  
embedding_dim=6                                                         
epoches=600
nc=100
num_day = 1     
data_name = 'ETTh1_3D'
data_pth='../matrix_filling_pq/data/ETTh1/ETTh1_3D.mat'
result_pth='../matrix_filling_pq/result 
save_dir_pth = '../matrix_filling_pq/save_model/NTF/ETTh1/1/'


train_NTF(lr,batch_size,sample_rate,embedding_dim,epoches,nc,num_day,data_name,data_pth,result_pth,save_dir_pth)
print("over")
        

            