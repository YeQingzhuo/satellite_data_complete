import torch as t
import torch.nn as nn
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
 
from utils.metrics import metric
import h5py
import time
import argparse
from model.utils_for_train import find_nearest_neighbors
from model.utils_for_train import loss_compare
from model.utils_for_train import get_score
from model.utils_for_train import get_dataslice
from model.utils_for_train import get_data_all
from model.utils_for_train import set_seed
from torch.utils.data import DataLoader
from data_provider.data_factory import data_provider
from utils.print_args import print_args
from exp.exp_imputation_3 import Exp_Imputation
from model.sample import random_missing_data
from model.sample import apply_missing_data
from utils.neighbor import find_nearest_neighbors
from model.utils_for_train import window_slide
from data_provider.data_factory import data_provider
from model.TimesNet import Model
import torch


 
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


def train_TimesNet( ): 
     
    parser = argparse.ArgumentParser(description='TimesNet')

     
    parser.add_argument('--task_name', type=str, required=True, default='imputation',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='TimesNet',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

     
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_mat_path', type=str, default='ETTh1_3D.mat', help='data file')
    parser.add_argument('--data_full_path', type=str, default='ETTh1_3D.mat', help='data file')
    parser.add_argument('--data_name', type=str, default='ETTh1', help='data name')
    parser.add_argument('--num_day', type=int, default=1, help='number of days')
    parser.add_argument('--window_size', type=int, default=1, help='window_size')
    parser.add_argument('--stride', type=int, default=1, help='stride')


    parser.add_argument('--result_path', type=str, default='1', help='result file')
    parser.add_argument('--save_dir_path', type=str, default='1', help='save model file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

     
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

     
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='sample ratio')
    parser.add_argument('--neighbor_num', type=int, default=10, help='number of neighbors')

     
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

     
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

     
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

     
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

     
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

     
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
     
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

     
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')


    try:
        args = parser.parse_args()
    except SystemExit as e:
        print(f"Error: {e}")
        print("Arguments passed:", sys.argv)
        sys.exit(2)



     
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

     
     

    Exp = Exp_Imputation

    if args.is_training:
        for ii in range(args.itr):
             
            exp = Exp(args)   
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            

             
            with open(args.result_path,'a+') as f:
                print(' 
                print('',file=f)
                print('sample_rate:',args.sample_rate,file=f)
                print('seq_len:',args.seq_len,file=f)
                print('mask_rate:',args.mask_rate,file=f)
                print('sample_rate:',args.sample_rate,file=f)
                print('train_epochs:',args.train_epochs,file=f)
                print('batch_size:',args.batch_size,file=f)
                print('learning_rate:',args.learning_rate,file=f)
                print('',file=f)
                print('',file=f)
            
            
            model = Model(args).float()
            n_param = nn.Parameter(t.tensor(2, dtype=t.float32))
            n_param_1 = nn.Parameter(t.tensor(10, dtype=t.float32))
            params = list(model.parameters()) +[n_param]+[n_param_1]
            model_optim = optim.Adam(params, lr=args.learning_rate, betas=(0.9,0.99))
             
             
             
             
             
            data_file_path =   os.path.join(args.root_path, args.data_mat_path)
             
            data_name = args.data_name
            m = scio.loadmat(data_file_path)
             
            data=t.Tensor(m[data_name]).to(device) 
             
            current_data_3D=data[:,:,-args.num_day:] 
            current_data = current_data_3D.reshape(-1, current_data_3D.size(1))   
            
            train_data1, train_loader1 = data_provider(args,flag='test1',epoch=0,mask=None)
            for epoch in range(args.train_epochs):
                 
                model.train() 
                print(f'epoch = [{epoch+1}/{args.train_epochs}]')         
                with open(args.result_path,'a+') as f:
                    print(f'epoch = [{epoch+1}/{args.train_epochs}]', file=f)
                 
                print('>>>>>>>start training>>>>>>>')
                running_loss_N=0
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,mask2,data2) in enumerate(train_loader1):
                 
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)

                     
                    B, T, N = batch_x.shape
                    mask1 = torch.rand((B, T, N)).to(device)
                    mask1[mask1 <= args.mask_rate] = 0   
                    mask1[mask1 > args.mask_rate] = 1   
                    mask2=torch.ones_like(batch_x).to(device)
                    mask2[batch_x==0]=0
                    inp = batch_x.masked_fill(mask1 == 0, 0)
                    mask3=torch.ones_like(batch_x).to(device)
                    mask3=(mask1 == 0) & (mask2 == 1)
                    outputs = model(inp, batch_x_mark, None, None, mask1)
                    loss = custom_loss(outputs[mask3 == 1], batch_x[mask3 == 1])
                    if loss.numel() > 0:
                            running_loss_N += torch.sum(loss)/loss.numel()
                    loss=torch.sum(loss)/loss.numel()
                    
                    loss.backward()
                        
                     
                    model_optim.step()
                     
                     
                     
                    
                    model_optim.zero_grad()
                sum_loss_N=0
                sum_loss_N = running_loss_N/len(train_loader1)
                with open(args.result_path,'a+') as f:
                    print('loss:',sum_loss_N,file=f) 
                    
                 
                mask = torch.zeros_like(current_data).to(device)     
                 
                sample_rate=args.sample_rate
                i=int(sample_rate*10)
                seed_num = epoch+1
                index_train,index_test= random_missing_data(mask,seed_num,i)
                index_train_2d = np.unravel_index(index_train, mask.shape)
                index_test_2d = np.unravel_index(index_test, mask.shape)
                mask[index_train_2d] = 1
                mask[index_test_2d] = 0               
                
                    
                 
                print('>>>>>>>start evaluating>>>>>>>') 
                 

                 
                 
                 
                 
                 
                 
                 
                train_data, train_loader = data_provider(args,flag='train',epoch=epoch,mask=mask)
                test_steps=len(train_loader) 

                sum_er = 0
                sum_mae = 0
                sum_mse = 0
                sum_rse = 0
                sum_rmse = 0
                sum_mape = 0
                sum_mspe = 0
                model.eval()

                 
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,mask2,data3) in enumerate(train_loader):
                    
                    batch_x_mark = batch_x_mark.float().to(device)

                     
                     
                     
                     
                    mask3 = torch.ones_like(batch_x)
                    mask3=mask3.float().to(device)
                    mask3[torch.abs(batch_x) == 0.0] = 0
                    batch_x = batch_x.float().to(device)
                    mask2 = mask2.float().to(device)
                    batch_x1 = batch_x.masked_fill(mask2 == 0, 0)
                    batch_x1 = batch_x1.float().to(device)
                    batch_x = batch_x.float().to(device)
                     
                     
                    outputs = model(batch_x1, batch_x_mark, None, None, mask2)
                    mask4 = mask2.float().to(device)

                    mask5=(mask4 == 0) & (mask3 == 1)
                    loss = custom_loss(outputs[mask5==1] , batch_x[mask5==1] )
                     
                     
                    
                    
                    pred = outputs.clone()   
                     
                    pred = pred*mask5
                      
                    true = batch_x.clone()   
                    true=true*mask5
                     
                     
                     
                     
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

                 
                with open(args.result_path,'a+') as f:
                    print('sample_rate:',args.sample_rate,file=f)
                     
                    print('test_er:',sum_er.item(),file=f)
                    print('test_mae:',sum_mae.item(),file=f)
                    print('test_mse:',sum_mse.item(),file=f)
                    print('test_rse:',sum_rse.item(),file=f)
                    print('test_rmse:',sum_rmse.item(),file=f)
                    print('test_mape:',sum_mape.item(),file=f)
                    print('test_mspe:',sum_mae.item(),file=f)
                    print('',file=f)


           
            



train_TimesNet()
print("over")
        

            