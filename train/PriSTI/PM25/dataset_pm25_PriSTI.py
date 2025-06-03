import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
import math


def get_missed_data(observed_values, missing_ratio, missing_type="random"):
    origin_masks = np.array(observed_values, dtype=bool).astype("int64")      
    observed_masks = origin_masks.copy()    
      
    for i in range(len(observed_values)):    
          
        obs_indices = np.where(origin_masks[i])[0].tolist()   
        missing_num = int(len(obs_indices) * missing_ratio)   
        if missing_num < 1:
            missing_num = 1
        elif missing_num > len(obs_indices) - 4:      
            missing_num = len(obs_indices) - 4
        observed_masks[i] = get_missed_mask(origin_masks[i], missing_num, missing_type)   

    observed_values = observed_values * observed_masks
    print("结束样本检查")

    return observed_values, observed_masks


def get_missed_mask(target_masks, missing_num, missing_type="random"):
      
    masks = target_masks.reshape(-1).copy().astype("int64")    
    masks_shape = masks.reshape(target_masks.shape)    

    obs_indices = np.where(masks)[0].tolist()    
    if missing_type == "random":
        miss_indices = np.random.choice(obs_indices, int(missing_num), replace=False).astype(
            "int64")    
    elif missing_type == "blackout":    
        miss_indices = obs_indices[int(len(obs_indices) - int(missing_num)):]

    masks[miss_indices] = False   
    missed_masks = masks.reshape(masks_shape.shape)    

    return missed_masks





class Dataset(Dataset):
    def __init__(self,current_day,num, eval_length=24, target_dim=36, val_len=0.1, is_interpolate=True,use_index_list=None,
                 mask_sensor=None, missing_ratio=None,matching_times=5, missing_type="random",mode="train",seed=1):
        self.eval_length = eval_length
        self.target_dim = target_dim

        self.is_interpolate = is_interpolate
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor
        self.mode = mode



          
        self.observed_data = np.zeros((350,24,36))  
        self.observed_masks = []  
        self.origin_values = []
        self.origin_masks = []  
        self.matched_values = []
        self.matched_dist = []
        self.matched_similarity_score = []
        self.missing_type = missing_type
        self.current_day = current_day
        self.num = num
        self.missing_type = missing_type

        path = ("./data/PM25/pm25_350.pk")
        with open(path, "rb") as f:  
                self.observed_values, self.observed_masks = pickle.load(f)

          
        self.current_values = self.observed_values[self.current_day:self.current_day+self.num,:,:]    
        self.current_masks = self.observed_masks[self.current_day:current_day+self.num:,:,:]   

          
        self.origin_values = self.current_values    
        self.origin_masks = np.array(self.origin_values, dtype=bool).astype("int64")

          
        self.current_values, self.current_masks = get_missed_data(self.current_values, missing_ratio, missing_type)    
        
          
        self.matched_values = self.current_values
        self.matched_dist = np.zeros((len(self.current_values), int(matching_times)))
        self.matched_similarity_score = np.zeros((len(self.current_values), int(matching_times)))
        self.matching_times = 0

          
        if use_index_list is None:    
            self.use_index_list = np.arange(len(self.current_values))
        else:
            self.use_index_list = use_index_list    

    def get_randmask(self, observed_mask, missing_type):    
        observed_mask = torch.tensor(observed_mask)
        observed_mask = observed_mask.float()
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask    
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)    
        observed_reverse_mask = observed_mask.permute(1,0).clone()    
        observed_for_mask = observed_reverse_mask.reshape(len(observed_mask), -1)    
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()    
            num_masked = math.floor(num_observed * sample_ratio)    

            if missing_type == "blackout" and self.use_hist_condition is True:
                ones_indices = (observed_for_mask[i] == 1).nonzero(as_tuple=True)
                indices_to_change = ones_indices[0][-num_masked:]     
                observed_for_mask[i][indices_to_change] = 0    
            else:
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1    
        if missing_type == "blackout" and self.use_hist_condition is True:
            rand_mask = observed_for_mask.reshape(observed_reverse_mask.shape).float()
            rand_mask = rand_mask.permute(1,0)    
        else:
            rand_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()    
        return rand_mask    


    def __getitem__(self, org_index):

        if self.current_values.shape[0]>=2:
            self.use_index_list=torch.arange(0, self.current_values.shape[0])
          
        index = self.use_index_list[org_index]
          
        
        s = {
            "observed_data": self.current_values[index],
            "observed_mask": self.current_masks[index],
            "origin_data": self.origin_values[index],
            "gt_mask": self.origin_masks[index],
            "matched_data": self.matched_values[index],
            "timepoints": np.arange(self.eval_length),    
            "matched_dist": self.matched_dist[index],
            "matched_similarity_score": self.matched_similarity_score[index],
            "matching_times": self.matching_times,
            "index": index,
        }

        observed_mask = s["observed_mask"]

        if self.mode != 'train':
            rand_mask = observed_mask
        else:
            rand_mask = self.get_randmask(observed_mask, self.missing_type)    
        
          
          
          
          
          
          
          
                
        s["cond_mask"] = rand_mask

        if self.is_interpolate:
            cond_mask = torch.tensor(rand_mask)
            tmp_data = torch.tensor(s["observed_data"]).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()

        return s

    def _values(self):
        return self.current_values
    def __len__(self):
        return len(self.use_index_list)



class neighbor_Dataset(Dataset):
    def __init__(self,eval_length=24, target_dim=36, val_len=0.1, is_interpolate=True,use_index_list=None,
                 mask_sensor=None, missing_ratio=None,matching_times=5, missing_type="random",mode="train",seed=1,neighbor_day=None,num=None):
        self.eval_length = eval_length
        self.target_dim = target_dim

        self.is_interpolate = is_interpolate
        self.missing_ratio = missing_ratio
        self.mask_sensor = mask_sensor
        self.mode = mode



          
        self.observed_data = np.zeros((350,24,36))  
        self.observed_masks = []  
        self.origin_values = []
        self.origin_masks = []  
        self.matched_values = []
        self.matched_dist = []
        self.matched_similarity_score = []
        self.missing_type = missing_type
        self.neighbor_day = neighbor_day
        self.num = num
        self.missing_type = missing_type

        path = ("./data/PM25/pm25_350.pk")
        with open(path, "rb") as f:  
                self.observed_values, self.observed_masks = pickle.load(f)

          
          
        self.current_values = self.observed_values[neighbor_day:neighbor_day+num,:,:]    
        self.current_masks = self.observed_masks[neighbor_day:neighbor_day+num:,:,:]   

          
        self.origin_values = self.current_values    
        self.origin_masks = np.array(self.origin_values, dtype=bool).astype("int64")

          
        self.current_values, self.current_masks = get_missed_data(self.current_values, missing_ratio, missing_type)    
        
          
        self.matched_values = self.current_values
        self.matched_dist = np.zeros((len(self.current_values), int(matching_times)))
        self.matched_similarity_score = np.zeros((len(self.current_values), int(matching_times)))
        self.matching_times = 0

          
        if use_index_list is None:    
            self.use_index_list = np.arange(len(self.current_values))
        else:
            self.use_index_list = use_index_list    

    def get_randmask(self, observed_mask, missing_type):    
        observed_mask = torch.tensor(observed_mask)
        observed_mask = observed_mask.float()
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask    
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)    
        observed_reverse_mask = observed_mask.permute(1,0).clone()    
        observed_for_mask = observed_reverse_mask.reshape(len(observed_mask), -1)    
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()    
            num_masked = math.floor(num_observed * sample_ratio)    

            if missing_type == "blackout" and self.use_hist_condition is True:
                ones_indices = (observed_for_mask[i] == 1).nonzero(as_tuple=True)
                indices_to_change = ones_indices[0][-num_masked:]     
                observed_for_mask[i][indices_to_change] = 0    
            else:
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1    
        if missing_type == "blackout" and self.use_hist_condition is True:
            rand_mask = observed_for_mask.reshape(observed_reverse_mask.shape).float()
            rand_mask = rand_mask.permute(1,0)    
        else:
            rand_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()    
        return rand_mask    


    def __getitem__(self, org_index):

        if self.current_values.shape[0]>=2:
            self.use_index_list=torch.arange(0, self.current_values.shape[0])
          
        index = self.use_index_list[org_index]
          
        
        s = {
            "observed_data": self.current_values[index],
            "observed_mask": self.current_masks[index],
            "origin_data": self.origin_values[index],
            "gt_mask": self.origin_masks[index],
            "matched_data": self.matched_values[index],
            "timepoints": np.arange(self.eval_length),    
            "matched_dist": self.matched_dist[index],
            "matched_similarity_score": self.matched_similarity_score[index],
            "matching_times": self.matching_times,
            "index": index,
        }

        observed_mask = s["observed_mask"]

        if self.mode == 'train':
            rand_mask = observed_mask
        else:
            rand_mask = self.get_randmask(observed_mask, self.missing_type)    
        cond_data = s["observed_data"] 
        cond_mask = rand_mask
        s["cond_mask"] = cond_mask

        if self.is_interpolate:
            cond_mask = torch.tensor(rand_mask)
            tmp_data = torch.tensor(s["observed_data"]).to(torch.float64)
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()

        return s

    def _values(self):
        return self.current_values
    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(current_day,num, device,seed=1, val_len=0.1, missing_ratio=0.1,batch_size=16,is_interpolate=True,
                    matching_times=5, missing_type="random", num_workers=4, mask_sensor=None):
    
    np.random.seed(seed)

    train_dataset = Dataset(current_day=current_day,num=num,missing_ratio=missing_ratio, seed=seed,
                                
                                   matching_times=matching_times, missing_type=missing_type , mode="train")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    test_dataset = Dataset(current_day=current_day,num=num,missing_ratio=missing_ratio, seed=seed,
                                  
                                  matching_times=matching_times, missing_type=missing_type , mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader,  test_loader ,train_dataset._values()


def get_dataloader_add_history(seed=1, val_len=0.1, missing_ratio=0.1,batch_size=16,is_interpolate=True,
                    matching_times=5, missing_type="random", num_workers=4, mask_sensor=None,neighbor_day=None,num = None):
    
    np.random.seed(seed)

    neighbor_dataset = neighbor_Dataset(missing_ratio=missing_ratio, seed=seed,
                                
                                   matching_times=matching_times, missing_type=missing_type , mode="train" , neighbor_day=neighbor_day,num = num)
    
    neighbor_loader = DataLoader(neighbor_dataset, batch_size=batch_size, shuffle=1)

    return neighbor_loader
    
    
