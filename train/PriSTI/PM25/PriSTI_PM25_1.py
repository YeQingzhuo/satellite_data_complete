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
from model.PriSTI import PriSTI_PM25
from dataset_pm25_PriSTI import get_dataloader
from model.utils_PriSTI import calc_quantile_CRPS , calc_quantile_CRPS_sum



parser = argparse.ArgumentParser(description="PriSTI")
parser.add_argument("--config", type=str, default="base_PriSTI.yaml")
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
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--matchingtimes", type=float, default=0)    
parser.add_argument("--learnhist", action="store_true")    
parser.add_argument("--dataset", type=str, default="pm25")
parser.add_argument("--missingtype", type=str, default="random")    
parser.add_argument("--useguide", type=str, default="True")  
parser.add_argument("--isadp", type=str, default="False") 
parser.add_argument("--adjfile", type=str, default="PM25") 




parser.print_help()
args = parser.parse_args()
print(args)


config_path = "config/" + args.config
with open(config_path, "r") as f:
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
config["model"]["use_guide"]=args.useguide
config["is_adp"]=args.isadp
config["diffusion"]["adj_file"] = 'PM25'



current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  
foldername = "./save_PriSTI/pm25_outsample" + str(args.nfold) + "_X1_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
  
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


  
device = torch.device("cuda:0")
print(torch.cuda.is_available())
    


  
seed_value = 42
set_seed(seed_value)

  
g = torch.Generator()
g.manual_seed(seed_value)

  
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


  
def train_PriSTI(lr,num_epoches,current_day,num,nsample,result_pth):

    model = PriSTI_PM25(config, args.device).to(args.device)    

      
    model=model.to(device).train()

      
    lr = lr
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.99))   

      
    num_epoches = num_epoches

    for epoch in range(num_epoches):

          
        optimizer.zero_grad()

        model.train()     
        
        result_pth = result_pth
        with open(result_pth,'a+') as f:
            print('epoch = ',epoch,file=f)
        
          
        print("............start training............")
        print("............X1............")
        current_day=current_day
        num=num
        train_loader,test_loader,train_dataset = get_dataloader(    
            current_day=current_day,
            num=num,
            seed=epoch,
              
            batch_size=config["train"]["batch_size"],
            missing_ratio=config["model"]["test_missing_ratio"],
              
              
              
            matching_times=config["model"]["matching_times"],
            missing_type=config["train"]["missing_type"],
            device = device
        )

        model.train()
        loss_sum = 0
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:    
            for batch_no, train_batch in enumerate(it, start=1):    
                optimizer.zero_grad()    

                missing_type = config["train"]["missing_type"]

                loss = model(train_batch)    
                  
                  
                  
                loss_sum += loss
                  
                  
                  

        loss_sum.backward() 
        optimizer.step()  
        with open(result_pth,'a+') as f:
                print("loss_total:", loss_sum.item(), file=f)

        
          
        print("............start evaluating single(only X1)............")
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
                    output = model.evaluate(test_batch,nsample)

                      
                    samples,  c_target, eval_points, observed_points, observed_time = output    
                      
                    samples = samples.permute(0, 1, 3, 2)    
                    c_target = c_target.permute(0, 2, 1)    
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                      
                    samples_median = samples.median(dim=1)
                    samples_final = samples_median.values
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

                
                RMSE = abs(rmse_total / evalpoints_total)
                MAE = abs(mae_total / evalpoints_total)
                ER = abs(er_total / evalpoints_total)
                RSE = abs(rse_total / evalpoints_total)
                MSE = abs(mse_total / evalpoints_total)
                MAPE = abs(mape_total / evalpoints_total)
                MSPE = abs(mspe_total / evalpoints_total)

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


       









lr = 0.000001
num_epoches = 1000
current_day=345   
num=1
nsample = 10     


result_pth = ' ../matrix_filling_pq/result  

train_PriSTI(lr=lr,num_epoches=num_epoches,current_day=current_day,num=num,nsample=nsample,result_pth=result_pth)
print("over")







