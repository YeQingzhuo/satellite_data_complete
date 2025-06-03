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
import time
from model.utils_for_train import find_nearest_neighbors
from model.utils_for_train import loss_compare
from model.utils_for_train import get_score
from model.utils_for_train import get_dataslice
from model.utils_for_train import get_data_all
from model.utils_for_train import get_sample


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



  
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true,W):
          
        loss = abs((y_pred-y_true))
        loss = loss*W
        return loss
  
custom_loss = CustomLoss()




def train_costco(sample_rate,embedding_dim,epoches,nc,count,Y,WX,pth): 
    batch_size = 1
    lr = 0.0001
    num_epoches = epoches
    sample_rate=sample_rate
    fz,fm=1,100000

    alpha=0.001
    beta=0.001
    ceta=0.00001

    mu=0
    sigma=1
    const=torch.Tensor([2]).to(device)
    
      
      
      
    m=scio.loadmat('../matrix_filling_pq/data/small_use_3D.mat')
      
      
      
    data=t.Tensor(m['small_data_part_3D']).to(device)    
    data=data[:,:,t.sum(data,(0,1))>=1]
    data=data[:,t.sum(data,(0,2))>=1,:]
    data=data[t.sum(data,(1,2))>1,:,:]
      
      
      
    data=data[-2:,:,:]
    
    mi = data.min()

      
    model=CostCo(data.shape[0],data.shape[1],data.shape[2],embedding_dim,nc)
    print(model)
    params=list(model.parameters())

      
    model=model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))   

      
      
    
    data_erf=data

      
    p=t.rand_like(data_erf)   
    Flag=True
    while Flag:
        tmp1=((p<=sample_rate) & (data!=0)).float()   
        for i in range(p.shape[2]):   
            if t.sum(tmp1[:,:,i])==0:   
                p=t.rand_like(data_erf)
                break

        if i==p.shape[2]-1:   
            Flag=not Flag
        else:
            pass

    for epoch in range(num_epoches):
        train_list,test_list=[],[]    
        model.train()     
        
        sum_loss_X1=0
        sum_loss_X2=0
        sum_loss = 0

        with open(pth,'a+') as f:
            print('epoch = ',epoch,file=f)
        
          
        
        print("............start training............")
        print("............X1............")
        for k in range(data_erf.shape[2]):    
              
            seed_num = int(epoch+1)
            traink,testk=get_sample(data_erf,k,sample_rate,seed_num)
            test_list+=testk
            train_list+=traink
            train_loader=t.utils.data.DataLoader(traink,batch_size=1,shuffle=True)    

              
            if (k+1)%5==0 and (k+1)>=5:
                print('[{}]/[{}]'.format(k+1,data_erf.shape[2]))
                   
            running_loss_X1=0
            
            time_train_start = time.time()
            
            for i,data_i in enumerate(train_loader,0):    
                o_inputs,d_inputs,t_inputs,scores= data_i
                scores=scores.float()
                t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)

                optimizer.zero_grad()     
                outputs = model(o_inputs, d_inputs, t_inputs).to(device)     
                outputs= torch.squeeze(outputs, 1)    
                
                  
                  
                loss = custom_loss(outputs, scores,WX)

                  
                loss.backward()
                optimizer.step()

                  
                running_loss_X1+=loss.item()
        
            sum_loss_X1+=running_loss_X1
                       
          
          
        print("............start evaluating single(only X1)............")
        time_evaluate_start = time.time() 
          
        if epoch>=0:
            model.eval()
            print('[{}]/[{}]'.format(epoch+1,num_epoches))
              
            train_eval=t.utils.data.DataLoader(train_list,batch_size=1,shuffle=True)
            test_eval=t.utils.data.DataLoader(test_list,batch_size=1,shuffle=True)

            test_er,test_mae,test_rmse,test_mape=get_score(dataset=test_eval,model=model,mi=mi,count=count,fz=fz,fm=fm)
            train_er,train_mae,train_rmse,train_mape=get_score(dataset=train_eval,model=model,mi=mi,count=count,fz=fz,fm=fm)

              
            with open(pth,'a+') as f:
                print('!!!!X1 evaluate result!!!!',file=f)
                print('sample_rate:',sample_rate,file=f)
                print('epoch:[{}]/[{}]'.format(epoch+1,num_epoches),file=f)
                print('nc:',nc,file=f)
                print('embedding_dim(R):',embedding_dim,file=f)
                print('test er:',test_er.item(),file=f)
                print('test_mae:',test_mae.item(),file=f)
                print('test_rmse:',test_rmse.item(),file=f)
                print('test_mape:',test_mape.item(),file=f)
                print('train_er:',train_er.item(),file=f)
                print('train_mae:',train_mae.item(),file=f)
                print('train_rmse:',train_rmse.item(),file=f)
                print('train_mape:',train_mape.item(),file=f)
                print('\n',file=f)

            time_evaluate_end = time.time()
            time_evaluate = time_evaluate_end - time_evaluate_start
            with open(pth,'a+') as f:
                print('evaluate time X1= ',time_evaluate,file=f)
            
            
        
          
        print("............X2............")
          
        model.train()     
        data_cpu = data.cpu()
        data_np = data_cpu.numpy()
        data_neigh = np.zeros_like(data_np)
        neigh_loader=t.utils.data.DataLoader(train_list,batch_size=1,shuffle=True)
        for i,data_i in enumerate(neigh_loader,0): 
            a,b,c,value= data_i
            data_neigh[a,b,c]=value
          
        neighbor_matrix,weight_neighbor = find_nearest_neighbors(data_neigh)
          
        for i in range(neighbor_matrix.shape[0]):
            WN = weight_neighbor[i]
            data_neigh_tensor = torch.tensor(neighbor_matrix[i,:,:,:]).to(device) 
            train_neigh_list=[]
            for k in range(data_neigh_tensor.shape[2]):
                if (k+1)%5==0 and (k+1)>=5:
                    print('[{}]/[{}]'.format(k+1,data_neigh_tensor.shape[2]))
                index_data_neigh=(data_neigh_tensor[:,:,k]!=0).nonzero()
                train_neigh=list(map(lambda x:(x[0],x[1],k,data[x[0],x[1],k]),index_data_neigh))    
                train_neigh_list+=train_neigh
                train_neigh_loader=t.utils.data.DataLoader(train_neigh,batch_size=1,shuffle=True)
                running_loss_X2=0
                for i,data_i in enumerate(train_neigh_loader,0):   
                    o_inputs,d_inputs,t_inputs,scores= data_i
                    scores=scores.float()
                    t_inputs, o_inputs, d_inputs, scores = t_inputs.to(device), o_inputs.to(device), d_inputs.to(device), scores.to(device)

                    optimizer.zero_grad()     
                    outputs = model(o_inputs, d_inputs, t_inputs)     
                    outputs= torch.squeeze(outputs, 1)    
     
                      
                    loss = custom_loss(outputs, scores,WN)

                  
                    loss.backward()
                    optimizer.step()

                  
                    running_loss_X2+=loss.item()
                sum_loss_X2+=running_loss_X2
        
        
        sum_loss=sum_loss_X1+sum_loss_X2
        print('[{}]/[{}], loss={:.4}'.format(epoch+1,num_epoches,sum_loss))
        time_train_end = time.time()
        time_train = time_train_end - time_train_start
        with open(pth,'a+') as f:
            print('train time = ',time_train,file=f)

        

          
          
        print("............start evaluating whole(X1+X2)............")
        time_evaluate_start = time.time() 
          
        if epoch>=0:
            model.eval()
            print('[{}]/[{}]'.format(epoch+1,num_epoches))
              
            train_eval=t.utils.data.DataLoader(train_list,batch_size=1,shuffle=True)
            test_eval=t.utils.data.DataLoader(test_list,batch_size=1,shuffle=True)

            test_er,test_mae,test_rmse,test_mape=get_score(dataset=test_eval,model=model,mi=mi,count=count,fz=fz,fm=fm)
            train_er,train_mae,train_rmse,train_mape=get_score(dataset=train_eval,model=model,mi=mi,count=count,fz=fz,fm=fm)

              
            with open(pth,'a+') as f:
                print('!!!!X1+X2 evaluate result!!!!',file=f)
                print('sample_rate:',sample_rate,file=f)
                print('epoch:[{}]/[{}]'.format(epoch+1,num_epoches),file=f)
                print('nc:',nc,file=f)
                print('embedding_dim(R):',embedding_dim,file=f)
                print('test er:',test_er.item(),file=f)
                print('test_mae:',test_mae.item(),file=f)
                print('test_rmse:',test_rmse.item(),file=f)
                print('test_mape:',test_mape.item(),file=f)
                print('train_er:',train_er.item(),file=f)
                print('train_mae:',train_mae.item(),file=f)
                print('train_rmse:',train_rmse.item(),file=f)
                print('train_mape:',train_mape.item(),file=f)
                print('\n',file=f)

            time_evaluate_end = time.time()
            time_evaluate = time_evaluate_end - time_evaluate_start
            with open(pth,'a+') as f:
                print('evaluate time X1= ',time_evaluate,file=f)
            
            