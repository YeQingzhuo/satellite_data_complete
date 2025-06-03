import argparse
import torch
import datetime
import json
import yaml
import os
import pickle
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
 
from model.Costco import CostCo
import h5py
import time
from model.utils_for_train import find_nearest_neighbors
from model.utils_for_train import loss_compare
from model.utils_for_train import get_score
from model.utils_for_train import get_dataslice
from model.utils_for_train import get_data_all
from model.utils_for_train import get_sample
from model.utils_for_train import create_3d_tensor_from_train_list
from model.utils_for_train import set_seed
from model.utils_for_train import normalize_sum_to_one
from model.utils_for_train import load_and_fill_data
from model.utils_for_train import window_slide
from tqdm import tqdm
from model.CSDI import CSDI_ETTh1
from dataset_ETTh1 import get_dataloader,get_dataloader_add_history
from model.utils_CSDI import calc_quantile_CRPS , calc_quantile_CRPS_sum



parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--testmissingratio", type=float, default=0.9)
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--usehistcondition", action="store_true")   
parser.add_argument("--diststrategy", type=str, default="L2")   
parser.add_argument("--matchstrategy", type=str, default="entire")   
parser.add_argument("--histproportion", type=float, default=0)   
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--matchingtimes", type=float, default=0)   
parser.add_argument("--learnhist", action="store_true")   
parser.add_argument("--dataset", type=str, default="ETTh1")
parser.add_argument("--missingtype", type=str, default="random")   




parser.print_help()
args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)


config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["model"]["use_hist_condition"] = args.usehistcondition
config["model"]["dist_strategy"] = args.diststrategy
config["model"]["match_strategy"] = args.matchstrategy
config["train"]["hist_proportion"] = args.histproportion
config["train"]["epochs"] = args.epochs
config["train"]["learn_hist"] = args.learnhist
config["model"]["matching_times"] = args.matchingtimes
config["train"]["missing_type"] = args.missingtype
config["train"]["batch_size"]=args.batch_size


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


foldername = "./save_CSDI/ETTh1_nfold" + str(args.nfold) + "_X1+X2_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
 
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


 
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









def train_CSDI_add_history(lr,num_epoches,current_day,num,nsample,WX,window_size,stride,data_name,result_pth,data_file_path,Normal,data_preprocess_full_pth,data_preprocess_full_name):
    model = CSDI_ETTh1(config, args.device).to(args.device)   

    model=model.to(device).train()
    lr = lr
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))  

    num_epoches = num_epoches
    for epoch in range(num_epoches):

         
        optimizer.zero_grad()

        
        model.train()    
        
        

        W=[]     
        L=[]     

        result_pth = result_pth
        with open(result_pth,'a+') as f:
            print('epoch = ',epoch,file=f)

        loss_total = 0
        
         
        print("............start training............")
        print("............X1............")
        current_day=current_day
        num=num
        train_loader,test_loader,train_dataset = get_dataloader(   
            current_day=current_day,
            num=num,
            seed=epoch,
            nfold=args.nfold,
            batch_size=config["train"]["batch_size"],
            missing_ratio=config["model"]["test_missing_ratio"],
            use_hist_cond=False,
            dist_strategy=config["model"]["dist_strategy"],
            match_strategy=config["model"]["match_strategy"],
            matching_times=config["model"]["matching_times"],
            missing_type=config["train"]["missing_type"],
        )

        loss_sum_1 = 0

        WX = WX

        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:   
            for batch_no, train_batch in enumerate(it, start=1):  
                optimizer.zero_grad()   

                missing_type = config["train"]["missing_type"]

                loss, index = model(train_batch, missing_type, is_train=1)   
                
                 
                 
                 
                 
                loss_sum_1 = loss_sum_1 + loss
                 
                if loss_sum_1 != loss_sum_1:   
                    print("avg_loss is NAN, index = ", index)
                     

         
         
         
         


        
         
        print("............X2............")
         
        current_data=train_dataset
         
         
        print(current_data.shape)
        data_neigh = current_data.transpose(2,1,0)
         
         
        data_file_path =  data_file_path
        data_preprocess_full_pth = data_preprocess_full_pth
         
        data_name = data_name
         
        start_row = 0
        end_row = current_day        
         

         
        data_preprocess_full_name = data_preprocess_full_name
        m2 = scio.loadmat(data_preprocess_full_pth)
        matrix_past_fill=t.Tensor(m2[data_preprocess_full_name]).to(device)
        matrix_past_fill = matrix_past_fill.cpu().numpy()

         
        window_size = window_size
        stride = stride
        time_sample_matrix_fill =window_slide(matrix_past_fill,current_day,window_size=window_size,stride=stride)
        time_sample_matrix_fill = time_sample_matrix_fill.astype(np.float32)
         
        m = scio.loadmat(data_file_path)
        matrix_past = m[data_name]
        matrix_past = matrix_past[:, :, start_row:end_row]
        time_sample_matrix=window_slide(matrix_past,current_day,window_size=window_size,stride=stride)
        time_sample_matrix = time_sample_matrix.astype(np.float32)


         
        neighbor_matrix,weight_neighbor,topk_idxs = find_nearest_neighbors(data_neigh,time_sample_matrix,time_sample_matrix_fill)

        
        topk_idxs = topk_idxs*stride
            
         
        loss_sum_2 = 0
        loss_sum_2_list = []
        for n in range(neighbor_matrix.shape[0]):
             
            neighbor_loader = get_dataloader_add_history(  
            seed=epoch,
            nfold=args.nfold,
            batch_size=config["train"]["batch_size"],
            missing_ratio=config["model"]["test_missing_ratio"],
            use_hist_cond=False,
            dist_strategy=config["model"]["dist_strategy"],
            match_strategy=config["model"]["match_strategy"],
            matching_times=config["model"]["matching_times"],
            missing_type=config["train"]["missing_type"],
            neighbor_day=topk_idxs[n],
            num=num
        )
            
             
            model.train()
            with tqdm(neighbor_loader, mininterval=5.0, maxinterval=50.0) as it:   
                for batch_no, train_batch in enumerate(it, start=1):  
                    optimizer.zero_grad()   

                    missing_type = config["train"]["missing_type"]

                    loss, index = model(train_batch, missing_type, is_train=1)   
                     
                     
                    loss_sum_2 += loss
                     
                     
                    if loss_sum_2 != loss_sum_2:   
                        print("avg_loss is NAN, index = ", index)
                         
                    
            loss_sum_2_list.append(loss_sum_2)
        
        W.append(WX)
        L.append(loss_sum_1)

        for i in range(len(loss_sum_2_list)):
            W.append(weight_neighbor[i])
            L.append(loss_sum_2_list[i])

        if Normal:
            W = normalize_sum_to_one(W)

        for i in range(len(W)):
            loss_total += L[i]* W[i]

        loss_total.backward()
        
         
         
         
        optimizer.step()   
        with open(result_pth,'a+') as f:
                print("loss_total:", loss_total.item(), file=f)

            
            
         
        print("............start evaluating single(X1 and X2)............")
        with torch.no_grad():
            model.eval()
            mse_total = 0
            mae_total = 0
            er_total = 0
            rse_total = 0
            rmse_total = 0
            mape_total = 0
            mspe_total = 0
            evalpoints_total = 0

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            hist_proportion = args.histproportion
            print("\n evaluate is started, and hist_proportion = ", hist_proportion, ",the args.histproportion = ", args.histproportion)
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    nsample = nsample
                    output = model.evaluate(test_batch, hist_proportion, nsample)

                     
                    samples, samples_final, c_target, eval_points, observed_points, observed_time = output   
                     
                    samples = samples.permute(0, 1, 3, 2)   
                    samples_final = samples_final.permute(0, 2, 1)
                    c_target = c_target.permute(0, 2, 1)   
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                     
                     
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    scaler = 1
                    mean_scaler=0

                     
                    mse_current = ((torch.abs((samples_final - c_target)) * eval_points) ** 2) * (scaler ** 2)
                    mae_current = (torch.abs((samples_final - c_target) * eval_points)) * scaler
                    er_current = torch.abs((samples_final - c_target)) * eval_points * scaler
                    rmse_current = torch.sqrt(mse_current)
                     
                    c_target_safe = torch.where(c_target == 0, torch.tensor(1e-8), c_target)
                    rse_current = (torch.abs((samples_final - c_target_safe) ** 2) / (c_target_safe ** 2) * eval_points) * (scaler ** 2)
                    mape_current = (torch.abs((samples_final - c_target_safe) / c_target_safe) * eval_points) * scaler * 100
                    mspe_current = ((torch.abs((samples_final - c_target_safe) / c_target_safe) ** 2) * eval_points) * (scaler ** 2) * 100
                    
                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    er_total += er_current.sum().item()
                    rse_total += rse_current.sum().item()
                    rmse_total += rmse_current.sum().item()
                    mape_total += mape_current.sum().item()
                    mspe_total += mspe_current.sum().item()

                    evalpoints_total += eval_points.sum().item()
                    print(f'evalpoints_total: {evalpoints_total}')
                     
                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )
                
                 
                with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                    all_target = torch.cat(all_target, dim=0)   
                    all_evalpoint = torch.cat(all_evalpoint, dim=0)   
                    all_observed_point = torch.cat(all_observed_point, dim=0)
                    all_observed_time = torch.cat(all_observed_time, dim=0)
                    all_generated_samples = torch.cat(all_generated_samples, dim=0)   
                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            scaler,   
                            mean_scaler,   
                        ],
                        f,
                    )

                CRPS = calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )
                CRPS_sum = calc_quantile_CRPS_sum(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )

                
                RMSE = rmse_total / evalpoints_total
                MAE = mae_total / evalpoints_total
                ER = er_total / evalpoints_total
                RSE = rse_total / evalpoints_total
                MSE = mse_total / evalpoints_total
                MAPE = mape_total / evalpoints_total
                MSPE = mspe_total / evalpoints_total

                with open(result_pth,'a+') as f:
                    print(args)
                    print(args, file=f)
                    print("testmissingratio:", args.testmissingratio, file=f)
                    print('test er:',ER,file=f)
                    print('test_mae:',MAE,file=f)
                    print('test_mse:',MSE,file=f)
                    print('test_rse:',RSE,file=f)
                    print('test_rmse:',RMSE,file=f)
                    print('test_mape:',MAPE,file=f)
                    print('test_mspe:',MSPE,file=f)
                    print("CRPS_sum:", CRPS_sum, file=f)
                     
                    print('\n',file=f)



lr = 0.00001
num_epoches = 500
current_day=724      
num=1
window_size = 1
stride = 1

nsample = 100    

WX = 1

Normal = True

data_name = 'ETTh1_3D'
data_file_path =  '../matrix_filling_pq/data/ETTh1/ETTh1_3D.mat'

result_pth = '../matrix_filling_pq/result 

data_preprocess_full_pth = '../matrix_filling_pq/data/data_preprocessing/ETTh1_full.mat'
data_preprocess_full_name = 'data'

train_CSDI_add_history(lr,num_epoches,current_day,num,nsample,WX,window_size,stride,data_name,result_pth,data_file_path,Normal,data_preprocess_full_pth,data_preprocess_full_name)
print('over')
