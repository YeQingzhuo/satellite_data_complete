from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import scipy.io as scio
import time
import warnings
import numpy as np

from utils.neighbor import find_nearest_neighbors
from model.utils_for_train import load_and_fill_data
from model.utils_for_train import window_slide
from model.utils_for_train import normalize_sum_to_one

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
    

custom_loss = CustomWeightedLoss()



class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag,epoch=0,begin_row=0):
        data_set, data_loader = data_provider(self.args, flag,epoch=epoch,begin_row=0)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9,0.99))
        return model_optim


    def train(self, setting):
        time_now = time.time()

        model_optim = self._select_optimizer()

        os.makedirs(self.args.save_dir_path, exist_ok=True)  

        

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            W=[]
            L=[]

            with open(self.args.result_path,'a+') as f:
                print('epoch = ',epoch,file=f)

            # ###########################train###########################
            train_loss = []
            train_data, train_loader = self._get_data(flag='train',epoch=epoch)

            train_steps = len(train_loader)

            model_optim.zero_grad()

            self.model.train()
            epoch_time = time.time()
            running_loss_X1 = 0
            sum_loss_X1 = 0

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # set mask 
                mask = torch.ones_like(batch_x).to(self.device)
                # epsilon = 1e-6  
                mask[torch.abs(batch_x) == 0.0] = 2  
                
                non_zero_indices = torch.nonzero(batch_x != 0.0, as_tuple=False)
                
                num_to_mask = int(self.args.mask_rate * non_zero_indices.size(0))   
                mask_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_to_mask]]
                
                mask[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]] = 0
                
                mask[(mask != 0) & (mask != 2)] = 1

                mask1=mask.clone()
                mask1[batch_x == 0.0] = 0  
                
                batch_x1 = batch_x * mask1   
                
                outputs = self.model(batch_x1, batch_x_mark, None, None, mask1,is_history=False)

                loss = custom_loss(outputs[mask == 0], batch_x[mask == 0],W,L)

                if loss.numel() > 0:
                    running_loss_X1 += torch.sum(loss)/loss.numel()
            
            sum_loss_X1= running_loss_X1 /train_steps

            train_loss.append(sum_loss_X1.item())

            # sum_loss_X1.backward()
            # model_optim.step()



            # ###########################find nearest neighbor train###########################
                

            current_data = train_data.data_x    
           
            current_data_tensor = torch.tensor(current_data).to(device)
            current_data_3D = current_data_tensor.unsqueeze(2)

            
            data_file_path =  os.path.join(self.args.root_path, self.args.data_mat_path)
            data_name = self.args.data_name
            m = scio.loadmat(data_file_path)                         
            data=torch.Tensor(m[data_name]).to(device) 
            datal=data
            
            start_row = 0
            end_row = datal.shape[2]-self.args.num_day       
            matrix_past_fill = load_and_fill_data(data_file_path, data_name, start_row, end_row)
            
            time_sample_matrix_fill =window_slide(matrix_past_fill,datal.shape[2],window_size=self.args.window_size,stride=self.args.stride)
            time_sample_matrix_fill = time_sample_matrix_fill.astype(np.float32)
            
            matrix_past = m[data_name]
            matrix_past = matrix_past[:, :, start_row:end_row]
            time_sample_matrix=window_slide(matrix_past,datal[2],window_size=self.args.window_size,stride=self.args.stride)
            time_sample_matrix = time_sample_matrix.astype(np.float32)
            
            current_data_3D =  current_data_3D.cpu().numpy()
            begin_row,neighbor_matrix,weight_neighbor = find_nearest_neighbors(current_data_3D,time_sample_matrix,time_sample_matrix_fill)
            
            sum_loss_X2=[]

            
            for n in range(neighbor_matrix.shape[0]):
                weight = weight_neighbor[n]  
                current_row = begin_row[n]  

                neighbor_data, neighbor_loader = self._get_data(flag='neighbor',epoch=epoch,begin_row=current_row)
                
                
                neighbor_steps = len(neighbor_loader)

                
                epoch_time = time.time()
                running_loss_X2_N = 0

                sum_loss_X2_N = 0

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(neighbor_loader):
                    iter_count += 1
                    

                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    
                    mask = torch.ones_like(batch_x).to(self.device)
                    
                    mask[torch.abs(batch_x) == 0.0] = 2  
                   
                    non_zero_indices = torch.nonzero(batch_x != 0.0, as_tuple=False)
                    
                    num_to_mask = int(self.args.mask_rate*2 * non_zero_indices.size(0))   
                    mask_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:num_to_mask]]
                    
                    mask[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]] = 0
                    
                    mask[(mask != 0) & (mask != 2)] = 1

                    mask1=mask.clone()
                    mask1[batch_x == 0.0] = 0  

                    batch_x1 = batch_x * mask1   
                 
                    outputs = self.model(batch_x1, batch_x_mark, None, None, mask1,is_history=True)

                    loss = custom_loss(outputs[mask == 0], batch_x[mask == 0],W,L)

                    if loss.numel() > 0:
                        running_loss_X2_N += torch.sum(loss)/loss.numel()
                
                sum_loss_X2_N= running_loss_X2_N /neighbor_steps

                sum_loss_X2.append(sum_loss_X2_N)

            # ————————sum loss————————
            
            WX=0.0
            W.append(WX)
            L.append(sum_loss_X1)
            weight_neighbor=weight_neighbor.numpy()
            for i in range(neighbor_matrix.shape[0]):
                W.append(weight_neighbor[i])
                L.append(sum_loss_X2[i])

            W = normalize_sum_to_one(W)
            sum_loss = custom_loss(y_pred=0,y_true=0,W=W,L=L)


            sum_loss.backward()
            model_optim.step()

                
            with open(self.args.result_path,'a+') as f:
                print('epoch sum loss = ',sum_loss.item(),file=f)

            if (epoch + 1) % 100 == 0:
                file_path = os.path.join(self.args.save_dir_path, f'X1_model_epoch_{epoch + 1}.pth')
                torch.save(self.model.state_dict(), file_path)
                print(f"Model X1 saved at {file_path}")


            # ###########################test###########################
            self.test(setting,epoch=epoch)

    def test(self, setting, test=0,epoch=0):
        test_data, test_loader = self._get_data(flag='test',epoch=epoch)

        preds = []
        trues = []
        masks = []

        self.model.eval()

        test_steps = len(test_loader)

        sum_er = 0.0
        sum_mae = 0.0
        sum_mse = 0.0
        sum_rse = 0.0
        sum_rmse = 0.0
        sum_mape = 0.0
        sum_mspe = 0.0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # set mask   
                print(f'batch_x: {batch_x}')
                mask = torch.rand_like(batch_x).to(self.device)
                epsilon = 1e-6
                mask[torch.abs(batch_x) < epsilon] = 2  
                # mask[batch_x != 0] = torch.where(torch.rand_like(batch_x) < 0.5, 0, 1)

                # mask values based on mask rate：mask
                mask[batch_x != 0.0] = torch.where(mask[batch_x != 0] <= self.args.mask_rate, 0, mask[batch_x != 0])  # masked
                mask[batch_x != 0.0] = torch.where(mask[batch_x != 0] > self.args.mask_rate, 1, mask[batch_x != 0])  # remain
                mask1=mask.clone()
                mask1[batch_x == 0.0] = 0   
                batch_x1 = batch_x * mask1   

                # imputation
                outputs = self.model(batch_x1, batch_x_mark, None, None, mask1,is_history=False)

                # eval
                pred = outputs.clone()  # 张量
                pred = pred[mask == 0]  
                true = batch_x.clone()  # 张量
                true = true[mask == 0]
                er, mae, mse,rse, rmse, mape, mspe = metric(pred, true)

                sum_er += er
                sum_mae += mae 
                sum_mse += mse
                sum_rse += rse
                sum_rmse += rmse
                sum_mape += mape
                sum_mspe += mspe


        sum_er = sum_er / test_steps
        sum_mae = sum_mae / test_steps
        sum_mse = sum_mse / test_steps
        sum_rse = sum_rse / test_steps
        sum_rmse = sum_rmse / test_steps
        sum_mape = sum_mape / test_steps
        sum_mspe = sum_mspe / test_steps  

        # result save
        with open(self.args.result_path,'a+') as f:
            print('sample_rate:',self.args.sample_rate,file=f)
            #print('epoch:[{}]/[{}]'.format(epoch+1,num_epoches),file=f)
            print('test_er:',sum_er.item(),file=f)
            print('test_mae:',sum_mae.item(),file=f)
            print('test_mse:',sum_mse.item(),file=f)
            print('test_rse:',sum_rse.item(),file=f)
            print('test_rmse:',sum_rmse.item(),file=f)
            print('test_mape:',sum_mape.item(),file=f)
            print('test_mspe:',sum_mae.item(),file=f)
            print('',file=f)

        
